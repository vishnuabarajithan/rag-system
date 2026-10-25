from agents import llm
from agents.state import AgentState
from config.settings import settings

_SYSTEM = """You are a precise, grounded research assistant.
Answer questions ONLY using the provided context chunks.
Always cite sources using [N] notation that matches the chunk references given.
If the context is insufficient to answer, say so clearly — never hallucinate.
For tables: show calculation steps explicitly.
For images: refer to the image descriptions provided."""

_INTENT_HINTS = {
    "analytical": "Provide detailed analysis with explicit reasoning steps.",
    "summarize":  "Provide a concise structured summary with clear sections.",
    "compare":    "Compare using a structured format (table or bullet points).",
    "factual":    "Answer directly and concisely.",
}


def _build_user_message(state: AgentState) -> str:
    context = state.get("context_window", "")
    query = state["user_query"]
    intent = state.get("query_intent", "factual")
    hint = _INTENT_HINTS.get(intent, "")
    return f"Context:\n{context}\n\n---\nQuestion: {query}\n{hint}"


def run(state: AgentState) -> dict:
    answer_text, input_tokens, output_tokens = llm.chat(
        _SYSTEM, _build_user_message(state), max_tokens=settings.max_answer_tokens
    )
    confidence = min(1.0, len(state.get("top_chunks", [])) / settings.context_top_n)

    return {
        "final_answer": answer_text,
        "confidence_score": confidence,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def stream_generate(state: AgentState, usage: dict):
    """Stream the grounded answer token-by-token.

    Yields text deltas as they arrive. `usage` is a mutable dict that is
    populated with `input_tokens`/`output_tokens` from the stream's final usage
    chunk. Same prompt/system instruction as the non-streaming `run` above.
    """
    yield from llm.chat_stream(_SYSTEM, _build_user_message(state), settings.max_answer_tokens, usage)
