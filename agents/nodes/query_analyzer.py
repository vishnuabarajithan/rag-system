import json
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

_SYSTEM = """You are a query analysis expert for a RAG system.
Given a user question, return ONLY valid JSON with these keys:
- expanded_queries: list of 1-3 semantically diverse sub-queries that cover the question from different angles
- query_intent: one of [factual, analytical, summarize, compare]
- query_filters: dict of metadata filters explicitly mentioned (e.g. {"file_type": "pdf", "sheet_name": "Sales"}) — empty dict if none
- requires_table_reasoning: true if the question involves numbers, spreadsheets, or tabular data
- requires_image_understanding: true if the question involves visual content, charts, or diagrams"""


def run(state: AgentState) -> dict:
    query = state["user_query"]
    iteration = state.get("iteration_count", 0)

    relaxation = ""
    if iteration > 0:
        relaxation = (
            "\nNote: a previous retrieval attempt returned no results. "
            "Generate broader, more general sub-queries this time."
        )

    response = _get_client().models.generate_content(
        model=settings.routing_model,
        contents=f"Query: {query}{relaxation}",
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=512,
            response_mime_type="application/json",
        ),
    )

    try:
        analysis = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        analysis = {}

    return {
        "expanded_queries": analysis.get("expanded_queries", [query]),
        "query_intent": analysis.get("query_intent", "factual"),
        "query_filters": analysis.get("query_filters", {}),
        "requires_table_reasoning": analysis.get("requires_table_reasoning", False),
        "requires_image_understanding": analysis.get("requires_image_understanding", False),
        "iteration_count": iteration + 1,
    }
