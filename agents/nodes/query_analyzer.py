from agents import llm
from agents.state import AgentState

_SYSTEM = """You are a query analysis expert for a RAG system.
Given a user question, return ONLY valid JSON (no prose, no code fences) with these keys:
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

    text, _, _ = llm.chat(_SYSTEM, f"Query: {query}{relaxation}", max_tokens=512)
    analysis = llm.parse_json(text)

    expanded = analysis.get("expanded_queries")
    if not isinstance(expanded, list) or not all(isinstance(q, str) for q in expanded):
        expanded = []
    filters = analysis.get("query_filters")

    return {
        "expanded_queries": expanded or [query],
        "query_intent": analysis.get("query_intent", "factual"),
        "query_filters": filters if isinstance(filters, dict) else {},
        "requires_table_reasoning": analysis.get("requires_table_reasoning", False),
        "requires_image_understanding": analysis.get("requires_image_understanding", False),
        "iteration_count": iteration + 1,
    }
