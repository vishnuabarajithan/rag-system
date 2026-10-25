from unittest.mock import MagicMock

import pytest


def _hit(cid, score):
    return {
        "chunk_id": cid, "document_id": "d", "content": cid,
        "content_type": "text", "score": score, "metadata": {"filename": "f.pdf"},
    }


@pytest.fixture
def retriever(monkeypatch):
    from agents.nodes import retriever as mod

    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 768
    monkeypatch.setattr(mod, "_embedder", embedder)
    return mod


def test_scores_summed_across_subqueries_and_sorted(retriever, monkeypatch):
    results = {
        "q1": [_hit("a", 0.5), _hit("b", 0.4), _hit("c", 0.3)],
        "q2": [_hit("c", 0.5), _hit("d", 0.2)],
    }
    store = MagicMock()
    store.hybrid_search.side_effect = lambda **kw: results[kw["query_text"]]
    monkeypatch.setattr(retriever, "_store", store)
    monkeypatch.setattr(retriever.settings, "context_top_n", 2)

    out = retriever.run({"user_query": "q", "expanded_queries": ["q1", "q2"], "iteration_count": 0})

    ids = [c["chunk_id"] for c in out["raw_chunks"]]
    assert ids == ["c", "a", "b", "d"]              # c fused: 0.3 + 0.5
    assert out["raw_chunks"][0]["score"] == pytest.approx(0.8)
    assert [c["chunk_id"] for c in out["top_chunks"]] == ["c", "a"]
    assert out["retrieval_quality"] == "sufficient"  # classified on the full set
    assert out["iteration_count"] == 1


def test_does_not_mutate_store_results(retriever, monkeypatch):
    shared = [_hit("a", 0.5)]
    store = MagicMock()
    store.hybrid_search.side_effect = lambda **kw: shared
    monkeypatch.setattr(retriever, "_store", store)

    out = retriever.run({"user_query": "q", "expanded_queries": ["q1", "q2"], "iteration_count": 0})

    assert out["raw_chunks"][0]["score"] == pytest.approx(1.0)
    assert shared[0]["score"] == 0.5
    assert out["retrieval_quality"] == "insufficient"


def test_empty(retriever, monkeypatch):
    store = MagicMock()
    store.hybrid_search.return_value = []
    monkeypatch.setattr(retriever, "_store", store)

    out = retriever.run({"user_query": "q", "iteration_count": 2})
    assert out["raw_chunks"] == [] and out["top_chunks"] == []
    assert out["retrieval_quality"] == "empty"
    assert out["iteration_count"] == 3
