"""
OpenRouterEmbedder
==================
Embeddings via OpenRouter's OpenAI-compatible `/embeddings` endpoint, default
model `qwen/qwen3-embedding-8b`. Same `embed` / `embed_batch` interface as the
other embedders; no local model weights.

Qwen3-Embedding is asymmetric: queries get a one-line task instruction,
documents are embedded as-is. Its native vectors are 4096-dim and trained with
Matryoshka representation learning, so they are truncated to
`settings.embedding_dim` (768, keeping the Qdrant schema unchanged) and
re-normalized for COSINE distance.

The API key is read at call time, so importing this module needs no key.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from config.settings import settings

_URL = "https://openrouter.ai/api/v1/embeddings"
_MAX_BATCH = 64          # inputs per request
_RETRIES = 4             # on 429 / 5xx, with exponential backoff
_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages from the user's documents "
    "that answer it\nQuery: "
)

_http: httpx.Client | None = None


def _get_http() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=60.0)
    return _http


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class OpenRouterEmbedder:
    def __init__(self, model: str | None = None):
        self._model = model or settings.openrouter_embedding_model

    def _embed_inputs(self, inputs: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
        payload = {"model": self._model, "input": inputs}
        for attempt in range(_RETRIES + 1):
            resp = _get_http().post(_URL, json=payload, headers=headers)
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if retryable and attempt < _RETRIES:
                time.sleep(float(resp.headers.get("retry-after") or 2 ** attempt))
                continue
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter embeddings error {resp.status_code}: {resp.text[:300]}"
                )
            body = resp.json()
            if "data" not in body:  # errors can also arrive with a 200
                raise RuntimeError(f"OpenRouter embeddings error: {str(body)[:300]}")
            rows = sorted(body["data"], key=lambda d: d.get("index", 0))
            return [_normalize(r["embedding"][:settings.embedding_dim]) for r in rows]
        raise RuntimeError("OpenRouter embeddings: retries exhausted")

    def embed(self, text: str) -> list[float]:
        # Single text is treated as a retrieval query.
        return self._embed_inputs([_QUERY_INSTRUCTION + text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Batched texts are documents being ingested. Each request has ~1 s of
        # fixed round-trip cost, so send few large requests, several at once;
        # map() keeps results in input order.
        batches = [texts[i:i + _MAX_BATCH] for i in range(0, len(texts), _MAX_BATCH)]
        if len(batches) <= 1:
            return self._embed_inputs(batches[0]) if batches else []
        workers = max(1, min(settings.embed_concurrency, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(self._embed_inputs, batches)
        return [vec for batch in results for vec in batch]
