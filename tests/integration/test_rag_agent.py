"""
Integration test: runs the full LangGraph graph against a mock Qdrant and a
mock embedder. Makes live Qwen calls, so it requires HF_TOKEN.
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
                "metadata": {"filename": "report.pdf", "page_num": 3},
            }
        ]
        yield mock


@pytest.fixture
def mock_embed():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    with patch("agents.nodes.retriever._embedder", embedder):
        yield embedder


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
    assert "<think>" not in result["final_answer"]
    assert isinstance(result.get("citations"), list)
    assert result.get("retrieval_quality") in ("sufficient", "insufficient", "empty")
