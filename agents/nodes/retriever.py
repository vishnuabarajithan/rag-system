from agents.state import AgentState, RetrievedChunk
from vectorstore.qdrant_store import QdrantStore
from ingestion.embedders import get_embedder
from config.settings import settings

_embedder = get_embedder()
_store = QdrantStore()


def run(state: AgentState) -> dict:
    # Multi-query fusion: each expanded sub-query gets its own hybrid (RRF)
    # search; a chunk's scores are summed across sub-queries, so chunks that
    # several sub-queries agree on rank highest. This ordering is what picks
    # the context (there is no separate reranker).
    fused: dict[str, RetrievedChunk] = {}

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
            if r["chunk_id"] in fused:
                fused[r["chunk_id"]]["score"] += r["score"]
            else:
                fused[r["chunk_id"]] = RetrievedChunk(**r)

    all_chunks = sorted(fused.values(), key=lambda c: c["score"], reverse=True)

    if len(all_chunks) == 0:
        quality = "empty"
    elif len(all_chunks) < 3:
        quality = "insufficient"
    else:
        quality = "sufficient"

    return {
        "raw_chunks": all_chunks,
        "top_chunks": all_chunks[:settings.context_top_n],
        "retrieval_quality": quality,
        # Bound the insufficient/empty retry loops: each retrieval attempt
        # counts toward the iteration cap checked in _route_after_retrieval.
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
