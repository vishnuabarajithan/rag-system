from agents.state import AgentState, RetrievedChunk
from vectorstore.qdrant_store import QdrantStore
from ingestion.embedders.text_embedder import TextEmbedder
from config.settings import settings

_embedder = TextEmbedder()
_store = QdrantStore()


def run(state: AgentState) -> dict:
    all_chunks: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    collection = (state.get("collection_names") or ["default"])[0]

    for query in state.get("expanded_queries", [state["user_query"]]):
        dense_vec = _embedder.embed(query)
        results = _store.hybrid_search(
            collection=collection,
            dense_vector=dense_vec,
            query_text=query,
            top_k=settings.retriever_top_k,
            filters=state.get("query_filters", {}),
            filenames=state.get("source_filenames") or None,
        )
        for r in results:
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                all_chunks.append(RetrievedChunk(**r))

    if len(all_chunks) == 0:
        quality = "empty"
    elif len(all_chunks) < 3:
        quality = "insufficient"
    else:
        quality = "sufficient"

    return {
        "raw_chunks": all_chunks,
        "retrieval_quality": quality,
        # Bound the insufficient/empty retry loops: each retrieval attempt
        # counts toward the iteration cap checked in _route_after_retrieval.
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
