from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    SparseVector,
    PointStruct,
    Prefetch,
    FusionQuery,
    Fusion,
)
from vectorstore.collections import ensure_collection
from config.settings import settings


_shared_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """One shared client per process.

    In-process (local) mode keeps its data inside a single client instance, so
    the ingest pipeline and the retriever must share the same one. A persistent
    local path also can't be opened by two clients at once.
    """
    global _shared_client
    if _shared_client is None:
        if settings.qdrant_local_path == ":memory:":
            _shared_client = QdrantClient(location=":memory:")
        elif settings.qdrant_local_path:
            _shared_client = QdrantClient(path=settings.qdrant_local_path)
        else:
            _shared_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
    return _shared_client


class QdrantStore:
    def __init__(self):
        self._client = _get_client()

    def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        query_text: str,
        top_k: int = 20,
        filters: dict | None = None,
        filenames: list[str] | None = None,
    ) -> list[dict]:
        ensure_collection(self._client, collection)

        from vectorstore.sparse_index import bm25_encode
        sparse_indices, sparse_values = bm25_encode(query_text)

        conditions = []
        if filters:
            conditions += [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
        if filenames:
            # Restrict retrieval to a set of source files (checkbox include/exclude).
            conditions.append(
                FieldCondition(key="filename", match=MatchAny(any=filenames))
            )
        qdrant_filter = Filter(must=conditions) if conditions else None

        results = self._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=top_k * 2,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )

        return [
            {
                "chunk_id":     str(r.id),
                "document_id":  r.payload.get("document_id", ""),
                "content":      r.payload.get("content", ""),
                "content_type": r.payload.get("content_type", "text"),
                "score":        r.score,
                "metadata": {
                    k: v for k, v in r.payload.items()
                    if k not in ("content", "document_id", "content_type")
                },
            }
            for r in results.points
        ]

    def upsert(self, collection: str, points: list[dict]) -> None:
        ensure_collection(self._client, collection)

        qdrant_points = [
            PointStruct(
                id=p["id"],
                vector={
                    "dense":  p["dense"],
                    "sparse": SparseVector(
                        indices=p["sparse"]["indices"],
                        values=p["sparse"]["values"],
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=qdrant_points)

    def drop_collection(self, collection: str) -> None:
        """Delete an entire collection (used to reset a session). No-op if absent."""
        try:
            self._client.delete_collection(collection_name=collection)
        except Exception:
            pass

    def drop_collections_with_prefix(self, prefix: str) -> int:
        """Delete every collection whose name starts with `prefix`; returns the count."""
        names = [c.name for c in self._client.get_collections().collections
                 if c.name.startswith(prefix)]
        for name in names:
            self.drop_collection(name)
        return len(names)

    def delete_by_filename(self, collection: str, filename: str) -> None:
        from qdrant_client.models import FilterSelector
        self._client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
                )
            ),
        )
