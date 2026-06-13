"""
Integration test: runs the full LangGraph graph against a mock Qdrant.
Requires GOOGLE_API_KEY in environment.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_qdrant():
    with patch("vectorstore.qdrant_store.QdrantStore.hybrid_search") as mock:
        mock.return_value = [
            {
                "chunk_id":    "abc123",
                "document_id": "doc1",
                "content":     "The quarterly revenue was $4.2M in Q3 2024.",
                "content_type":"text",
                "score":       0.91,
                "rerank_score":None,
                "metadata": {"filename": "report.pdf", "page_num": 3},
            }
        ]
        yield mock


@pytest.fixture
def mock_embed():
    with patch("ingestion.embedders.text_embedder.TextEmbedder.embed") as mock:
        mock.return_value = [0.1] * 768
        yield mock


@pytest.mark.integration
def test_rag_graph_returns_answer(mock_qdrant, mock_embed):
    from agents.rag_agent import rag_graph

    result = rag_graph.invoke({
        "user_query":       "What was the Q3 revenue?",
        "session_id":       "test-session",
        "collection_names": ["default"],
        "iteration_count":  0,
        "messages":         [],
    })

    assert result.get("final_answer"), "Expected a non-empty answer"
    assert isinstance(result.get("citations"), list)
    assert result.get("retrieval_quality") in ("sufficient", "insufficient", "empty")
