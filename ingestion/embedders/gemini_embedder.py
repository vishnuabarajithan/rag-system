"""
GeminiEmbedder
==============
Embeddings via the Gemini API (`gemini-embedding-001`). Drop-in replacement for
TextEmbedder — same `embed` / `embed_batch` interface — but holds no local model
weights, so it needs no GPU and avoids the HF Hub download/warning.

The client is lazy-initialized (genai.Client validates GOOGLE_API_KEY at
construction), matching the pattern in agents/nodes/{generator,query_analyzer}.py
so that importing this module stays key-free for unit tests.

Output vectors are truncated to `settings.embedding_dim` (768) to keep the Qdrant
schema unchanged, then L2-normalized — Gemini only returns normalized vectors at
the native 3072 dim, and the dense vector is compared with COSINE distance.
"""
import math

from google import genai
from google.genai import types

from config.settings import settings

# Per-request cap on number of inputs; batch larger calls into chunks of this.
_MAX_BATCH = 100

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class GeminiEmbedder:
    def __init__(self, model: str | None = None):
        self._model = model or settings.gemini_embedding_model

    def _embed_contents(self, contents: list[str], task_type: str) -> list[list[float]]:
        resp = _get_client().models.embed_content(
            model=self._model,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )
        return [_normalize(list(e.values)) for e in resp.embeddings]

    def embed(self, text: str) -> list[float]:
        # Single text is treated as a retrieval query.
        return self._embed_contents([text], task_type="RETRIEVAL_QUERY")[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Batched texts are documents being ingested.
        out: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            out.extend(
                self._embed_contents(texts[i:i + _MAX_BATCH], task_type="RETRIEVAL_DOCUMENT")
            )
        return out
