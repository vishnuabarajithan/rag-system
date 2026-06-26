# CLAUDE.md
claude
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project runs from a project-local venv at `.venv` (Python 3.11). Use its interpreter explicitly on Windows — the base Anaconda Python is 3.7 and unusable here.

```bash
.venv\Scripts\python.exe -m pytest tests/unit -v                        # fast, no API key
.venv\Scripts\python.exe -m pytest tests/integration -v -m integration  # live Gemini call, needs GOOGLE_API_KEY
.venv\Scripts\python.exe -m pytest tests/unit/test_sparse_index.py -v   # a single test file
.venv\Scripts\python.exe -m pytest tests/unit/test_chunker.py::<name>   # a single test
```

Run the system locally with **no Docker** (embedded in-process Qdrant, persists to `./qdrant_local/`):

```bash
python webapp.py                  # FastHTML web chat UI → http://localhost:5001
python chat.py path/to/doc.pdf    # CLI: ingest a doc, then loop Q&A
python chat.py                    # CLI: skip ingest, query what's already stored
```

The full HTTP API stack (real Qdrant server, Redis, MLflow, Celery) needs Docker, which is **not installed** on this machine — `docker compose up --build` won't run without installing Docker Desktop first. API would serve Swagger at `:8000/docs`, MLflow at `:5000`.

## Architecture

A document QA system: a **LangGraph** agent answers questions grounded in ingested documents, using hybrid retrieval + cross-encoder reranking + **Google Gemini** generation. There are two execution paths that share the same core (`ingestion/`, `vectorstore/`, `agents/`):

- **Local path** (`chat.py`, `webapp.py`): embedded Qdrant, synchronous ingest, no external services.
- **Production path** (`api/`, `tasks/`): Qdrant server, Redis semantic cache, Celery async ingestion, MLflow tracing.

### Agent graph ([agents/rag_agent.py](agents/rag_agent.py))

`query_analyzer → retriever → (reranker → context_builder → generator) → END`, with a retry loop back into `query_analyzer` (on empty retrieval) or `retriever` (on insufficient). Each node is a pure `run(state) -> dict` function in [agents/nodes/](agents/nodes/) returning a partial `AgentState` update. State shape is the single source of truth in [agents/state.py](agents/state.py).

**Retry loop subtlety:** the loop is bounded by `iteration_count < 3` in `_route_after_retrieval`, but **both** `query_analyzer` and `retriever` increment `iteration_count` by 1, so a full empty-retry cycle advances it by 2. The retriever's own increment is what actually prevents the infinite loop — don't remove it. Retrieval quality is classified purely by chunk count: 0 → `empty`, <3 → `insufficient`, else `sufficient`.

### Vector store ([vectorstore/](vectorstore/))

Hybrid search = dense (`all-mpnet-base-v2`, 768-dim, cosine) + BM25 sparse, fused server-side via Qdrant RRF (`Prefetch` + `FusionQuery`, not dicts). Collections are auto-created lazily by `ensure_collection`.

Several invariants here are load-bearing and were the source of past bugs — preserve them:
- **One shared `QdrantClient` per process** (`_get_client` in [vectorstore/qdrant_store.py](vectorstore/qdrant_store.py)). In-process mode keeps data inside a single client instance and takes an exclusive file lock on `qdrant_local/`; ingest and retriever must use the same client, and two clients cannot open the path at once.
- **Chunk IDs are `uuid5`** of `filename-page-content[:64]` ([semantic_chunker.py](ingestion/chunkers/semantic_chunker.py) `_make_id`) — Qdrant point IDs must be UUID/uint, and uuid5 keeps identical content stable/idempotent.
- **BM25 buckets use `hashlib.blake2b`, not builtin `hash()`** ([vectorstore/sparse_index.py](vectorstore/sparse_index.py)) — builtin hash is salted per process, so ingest-time and query-time vectors wouldn't match. Token buckets are aggregated into a dict because Qdrant rejects sparse vectors with duplicate indices.

### Gemini clients

Generation and query-analysis nodes hold module-level Gemini clients that are **lazy-initialized** via a `_get_client()` global, because `genai.Client` validates `GOOGLE_API_KEY` at construction — eager init would break unit tests that have no key. Keep this pattern when adding Gemini-backed nodes. `query_analyzer` requests `response_mime_type="application/json"` and parses the result, defaulting gracefully on parse failure.

### Ingestion ([ingestion/pipeline.py](ingestion/pipeline.py))

`load → chunk → embed → upsert`, dispatched by file extension (PDF / spreadsheet / image / text). PDFs split text and tables separately; spreadsheets/DB tables emit a schema chunk plus 20-row windows. The DB loader strips credentials from the stored `filename`.

## Configuration

All settings are Pydantic `BaseSettings` in [config/settings.py](config/settings.py), loaded from `.env`. Key knobs: `GOOGLE_API_KEY` (required), `GENERATION_MODEL`/`ROUTING_MODEL` (`gemini-2.5-flash`), `RETRIEVER_TOP_K` (20), `RERANKER_TOP_N` (6), and `QDRANT_LOCAL_PATH` — set to a folder path for in-process Qdrant (`chat.py` sets this before importing any project module, which is required for the override to take effect) or `:memory:` for ephemeral.

## Gotchas

- `webapp.py` must run with `reload=False` (already set): the auto-reloader spawns a second process that collides with the embedded Qdrant file lock (`AlreadyLocked`). FastHTML `@rt(path)` registers GET+POST by default. If port 5001 won't rebind after a kill, an orphaned `multiprocessing.spawn` child may still hold the socket — find it via `Get-CimInstance Win32_Process` and `taskkill /F`.
- The current LLM provider is **Google Gemini** (`google-genai` SDK), migrated from Anthropic. Use `from google import genai`.
