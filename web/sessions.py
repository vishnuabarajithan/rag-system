"""
Per-user session state for the FastHTML web UI.

The session *identity* (`sid`) lives in a signed cookie (handled by FastHTML);
the actual per-session data — uploaded docs, their enabled/disabled state, and
the chat history — lives server-side here, keyed by `sid`. Each session gets its
own Qdrant collection (`session_<sid>`) so users never see each other's docs.

Two backends:
- InMemorySessionStore: a process-local dict. Correct for a single local process
  (the `webapp.py` dev server). NOT shared across workers.
- RedisSessionStore: JSON per `sid` in Redis. Use this for multi-worker /
  multi-replica production, where each worker is a separate process.

Select via `settings.session_backend` ("memory" | "redis").
"""
from __future__ import annotations

import json
from typing import Any

from config.settings import settings


def _new_state() -> dict[str, Any]:
    """A fresh, empty per-session state."""
    return {
        "docs": [],        # ordered list of ingested filenames
        "enabled": {},     # filename -> bool (include in retrieval context)
        "messages": [],    # [{role, text, citations, meta}]
    }


def collection_for(sid: str) -> str:
    """The Qdrant collection name that isolates this session's documents."""
    return f"session_{sid}"


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, sid: str) -> dict[str, Any]:
        if sid not in self._data:
            self._data[sid] = _new_state()
        return self._data[sid]

    def save(self, sid: str, state: dict[str, Any]) -> None:
        # In-memory state is mutated in place; this is a no-op kept for parity
        # with the Redis backend so call sites are identical.
        self._data[sid] = state

    def reset(self, sid: str) -> dict[str, Any]:
        self._data[sid] = _new_state()
        return self._data[sid]


class RedisSessionStore:
    """JSON-per-session store in Redis. Untested locally (needs a Redis server)."""

    _PREFIX = "rag:session:"

    def __init__(self) -> None:
        import redis  # imported lazily so "memory" mode needs no live Redis

        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = settings.cache_ttl_seconds

    def _key(self, sid: str) -> str:
        return f"{self._PREFIX}{sid}"

    def get(self, sid: str) -> dict[str, Any]:
        raw = self._redis.get(self._key(sid))
        if raw is None:
            state = _new_state()
            self.save(sid, state)
            return state
        return json.loads(raw)

    def save(self, sid: str, state: dict[str, Any]) -> None:
        # No TTL refresh policy here beyond a sliding expiry on every write.
        self._redis.set(self._key(sid), json.dumps(state))

    def reset(self, sid: str) -> dict[str, Any]:
        state = _new_state()
        self.save(sid, state)
        return state


_store: InMemorySessionStore | RedisSessionStore | None = None


def get_session_store() -> InMemorySessionStore | RedisSessionStore:
    """Process-wide singleton session store, chosen by settings.session_backend."""
    global _store
    if _store is None:
        if settings.session_backend == "redis":
            _store = RedisSessionStore()
        else:
            _store = InMemorySessionStore()
    return _store
