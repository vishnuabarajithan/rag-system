from agents.state import AgentState
from reranking.cross_encoder import CrossEncoderReranker
from config.settings import settings

_reranker = CrossEncoderReranker(model=settings.reranker_model)


def run(state: AgentState) -> dict:
    query = state["user_query"]
    chunks = state.get("raw_chunks", [])

    if not chunks:
        return {"reranked_chunks": [], "error": "No chunks available to rerank"}

    scored = _reranker.rerank(
        query=query,
        documents=[c["content"] for c in chunks],
        top_n=settings.reranker_top_n,
    )

    reranked = []
    for score, idx in scored:
        chunk = dict(chunks[idx])
        chunk["rerank_score"] = float(score)
        reranked.append(chunk)

    return {"reranked_chunks": reranked}
