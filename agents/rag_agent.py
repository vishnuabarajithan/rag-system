"""
LangGraph RAG Agent
===================
Graph topology:

    query_analyzer
          │
       retriever  ◄──────────────────────┐
          │                              │ (empty / insufficient)
          ├─ sufficient ──► reranker     │
          └─ empty/insufficient ─────────┘ (max 2 retries)
                                │
                         context_builder
                                │
                            generator
                                │
                              END

LangGraph concepts demonstrated:
  - StateGraph with typed AgentState
  - add_conditional_edges for routing
  - Self-loop retry for retrieval
  - Single END node
"""
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    query_analyzer,
    retriever,
    reranker,
    context_builder,
    generator,
)


def _route_after_retrieval(state: AgentState) -> str:
    quality = state.get("retrieval_quality", "empty")
    iteration = state.get("iteration_count", 0)

    if quality == "empty" and iteration < 3:
        return "query_analyzer"         # widen the query and retry
    if quality == "insufficient" and iteration < 3:
        return "retriever"              # re-run with expanded queries
    if state.get("error"):
        return END
    return "reranker"


def _route_after_rerank(state: AgentState) -> str:
    if state.get("error"):
        return END
    return "context_builder"


def build_rag_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_analyzer",  query_analyzer.run)
    graph.add_node("retriever",       retriever.run)
    graph.add_node("reranker",        reranker.run)
    graph.add_node("context_builder", context_builder.run)
    graph.add_node("generator",       generator.run)

    graph.set_entry_point("query_analyzer")

    graph.add_edge("query_analyzer", "retriever")

    graph.add_conditional_edges(
        "retriever",
        _route_after_retrieval,
        {
            "query_analyzer": "query_analyzer",
            "retriever":      "retriever",
            "reranker":       "reranker",
            END:              END,
        },
    )

    graph.add_conditional_edges(
        "reranker",
        _route_after_rerank,
        {
            "context_builder": "context_builder",
            END:               END,
        },
    )

    graph.add_edge("context_builder", "generator")
    graph.add_edge("generator",       END)

    return graph.compile()


def build_context_graph() -> StateGraph:
    """Same pipeline as `build_rag_graph` but stopping at `context_builder`
    (no generation). Used by the streaming web path: it produces the context
    window + citations, then generation is streamed token-by-token outside the
    graph. Reuses the exact same nodes and retry routing."""
    graph = StateGraph(AgentState)

    graph.add_node("query_analyzer",  query_analyzer.run)
    graph.add_node("retriever",       retriever.run)
    graph.add_node("reranker",        reranker.run)
    graph.add_node("context_builder", context_builder.run)

    graph.set_entry_point("query_analyzer")
    graph.add_edge("query_analyzer", "retriever")

    graph.add_conditional_edges(
        "retriever",
        _route_after_retrieval,
        {
            "query_analyzer": "query_analyzer",
            "retriever":      "retriever",
            "reranker":       "reranker",
            END:              END,
        },
    )
    graph.add_conditional_edges(
        "reranker",
        _route_after_rerank,
        {
            "context_builder": "context_builder",
            END:               END,
        },
    )
    graph.add_edge("context_builder", END)

    return graph.compile()


rag_graph = build_rag_graph()
context_graph = build_context_graph()
