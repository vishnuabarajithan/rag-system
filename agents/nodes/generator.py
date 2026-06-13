from google import genai
from google.genai import types
from agents.state import AgentState
from config.settings import settings

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client

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


def run(state: AgentState) -> dict:
    context = state.get("context_window", "")
    query = state["user_query"]
    intent = state.get("query_intent", "factual")
    hint = _INTENT_HINTS.get(intent, "")

    user_message = f"Context:\n{context}\n\n---\nQuestion: {query}\n{hint}"

    response = _get_client().models.generate_content(
        model=settings.generation_model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=4096,
        ),
    )

    answer_text = response.text or ""

    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    confidence = min(1.0, len(state.get("reranked_chunks", [])) / settings.reranker_top_n)

    return {
        "final_answer": answer_text,
        "confidence_score": confidence,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
