import math
import threading
import time

import pytest

from ingestion.embedders import openrouter_embedder as mod


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class _FakeHttp:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json, headers):
        self.calls.append(json)
        return self.responses.pop(0)


def _vec(n, dim=4096):
    return [float(n)] * dim


@pytest.fixture
def fake(monkeypatch):
    def install(*responses):
        http = _FakeHttp(responses)
        monkeypatch.setattr(mod, "_http", http)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        return http
    return install


def test_truncates_normalizes_and_orders(fake):
    http = fake(_Resp(200, {"data": [{"index": 1, "embedding": _vec(2)},
                                     {"index": 0, "embedding": _vec(1)}]}))
    out = mod.OpenRouterEmbedder().embed_batch(["a", "b"])
    assert len(out) == 2 and all(len(v) == mod.settings.embedding_dim for v in out)
    assert math.isclose(sum(x * x for x in out[0]), 1.0, rel_tol=1e-6)
    assert http.calls[0]["input"] == ["a", "b"]          # documents: no instruction


def test_query_gets_instruction(fake):
    http = fake(_Resp(200, {"data": [{"index": 0, "embedding": _vec(1)}]}))
    mod.OpenRouterEmbedder().embed("what is x")
    assert http.calls[0]["input"][0].startswith("Instruct:")
    assert http.calls[0]["input"][0].endswith("Query: what is x")


class _EchoHttp:
    """Answers each request from its own payload (thread-safe). Input "k" gets
    a one-hot vector at position k, so output order is checkable after
    normalisation. The first batch replies last, to exercise out-of-order
    completion."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def post(self, url, json, headers):
        with self._lock:
            self.calls.append(json)
        nums = [int(t) for t in json["input"]]
        time.sleep(0.1 if nums[0] == 0 else 0)
        return _Resp(200, {"data": [
            {"index": i, "embedding": [1.0 if d == n else 0.0 for d in range(4096)]}
            for i, n in enumerate(nums)
        ]})


def test_batches_large_inputs_in_parallel_and_keeps_order(monkeypatch):
    http = _EchoHttp()
    monkeypatch.setattr(mod, "_http", http)
    n = 2 * mod._MAX_BATCH + 1                     # 129 inputs → 3 requests
    out = mod.OpenRouterEmbedder().embed_batch([str(i) for i in range(n)])
    assert len(http.calls) == 3
    assert [v.index(1.0) for v in out] == list(range(n))


def test_retries_on_429_then_succeeds(fake):
    http = fake(_Resp(429, {"error": "slow down"}, {"retry-after": "1"}),
                _Resp(200, {"data": [{"index": 0, "embedding": _vec(1)}]}))
    assert len(mod.OpenRouterEmbedder().embed_batch(["a"])) == 1
    assert len(http.calls) == 2


def test_error_raises(fake):
    fake(_Resp(402, {"error": {"message": "Insufficient credits"}}))
    with pytest.raises(RuntimeError, match="402"):
        mod.OpenRouterEmbedder().embed_batch(["a"])


def test_error_body_with_200_raises(fake):
    fake(_Resp(200, {"error": {"message": "model not found"}}))
    with pytest.raises(RuntimeError, match="model not found"):
        mod.OpenRouterEmbedder().embed_batch(["a"])
