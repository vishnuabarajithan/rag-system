"""
Semantic cache backed by Redis.
Queries within cosine distance threshold return the cached response immediately,
skipping the full RAG pipeline and saving tokens + latency.
"""
import json
import numpy as np
import redis

from ingestion.embedders.text_embedder import TextEmbedder
from config.settings import settings

_CACHE_KEY_PREFIX = "rag:cache:"


class SemanticCache:
    def __init__(self):
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._embedder = TextEmbedder()

    def get(self, query: str) -> dict | None:
        query_vec = np.array(self._embedder.embed(query))

        for key in self._redis.scan_iter(f"{_CACHE_KEY_PREFIX}*"):
            raw = self._redis.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                cached_vec = np.array(entry["embedding"])
                similarity = float(
                    np.dot(query_vec, cached_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-9)
                )
                if similarity >= settings.cache_similarity_threshold:
                    return entry["response"]
            except Exception:
                continue

        return None

    def set(self, query: str, response: dict) -> None:
        embedding = self._embedder.embed(query)
        key = f"{_CACHE_KEY_PREFIX}{abs(hash(query))}"
        self._redis.setex(
            key,
            settings.cache_ttl_seconds,
            json.dumps({"embedding": embedding, "response": response}),
        )

    def invalidate_all(self) -> int:
        keys = list(self._redis.scan_iter(f"{_CACHE_KEY_PREFIX}*"))
        if keys:
            return self._redis.delete(*keys)
        return 0
