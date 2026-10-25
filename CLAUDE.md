# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project runs from a project-local venv at `.venv` (Python 3.11). Use its interpreter explicitly on Windows — the base Anaconda Python is 3.7 and unusable here.

```bash
.venv\Scripts\python.exe -m pytest tests/unit -v                        # fast, no API key
.venv\Scripts\python.exe -m pytest tests/integration -v -m integration  # live Qwen call, needs GROQ_API_KEY
.venv\Scripts\python.exe -m pytest tests/unit/test_sparse_index.py -v   # a single test file
.venv\Scripts\python.exe -m pytest tests/unit/test_chunker.py::<name>   # a single test
```

Run the system locally with **no Docker** (embedded in-process Qdrant, persists to `./qdrant_local/`):

```bash
python webapp.py                  # FastHTML web chat UI → http://localhost:5001
python chat.py path/to/doc.pdf    # CLI: ingest a doc, then loop Q&A
python chat.py                    # CLI: skip ingest, query what's already stored
```

`webapp.py` (FastHTML) is the single UI. Sessions are **ephemeral**: every load of `/` (refresh, or the redirect after Reset) discards the previous session's collection + state (`_discard_session`), and at startup in embedded mode all leftover `session_*` collections are dropped (never with a shared Qdrant server — other workers' live sessions would go). Citations: inline `[N]` markers in the final answer become links and only cited sources get chips; both open a `<dialog>` preview (`_link_citations`, `cite_chip`, `_CITE_JS`). It isolates each browser session: a signed-cookie `sid` maps to its own `session_<sid>` Qdrant collection, and per-session state (docs, per-source toggles, chat) lives in a `web/sessions.py` store — `InMemorySessionStore` by default (`SESSION_BACKEND=memory`), or `RedisSessionStore` (`SESSION_BACKEND=redis`) for multi-worker. For horizontally scaled production: point at a Qdrant **server** (leave `QDRANT_LOCAL_PATH` empty so `QDRANT_URL` is used — embedded Qdrant's exclusive single-process file lock forbids >1 worker), set `SESSION_BACKEND=redis` + a real `SESSION_SECRET`, and run `uvicorn webapp:app --workers N` (not the dev `serve()`).

### MLflow monitoring (works locally, no Docker)

Every `webapp.py` `/ask` is traced via `monitoring/mlflow_tracer.py` — `get_tracer()` returns a `RAGTracer` that logs per query: token counts (`input/output/total_tokens`), `total_latency_ms`, retrieval scores + chunk counts, `confidence_score`, `num_citations`, and the answer/citations as artifacts. Locally the tracking store defaults to a SQLite file (`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`), so no server is needed. To view the dashboard, run from the **project root** in a second terminal:

```bash
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # → http://localhost:5000
```

Both the app and the UI must use the same `sqlite:///mlflow.db` and run from the project root (relative path). Monitoring is **crash-proof and lazy**: it never connects at import, every MLflow call is exception-swallowed, and if the backend is unreachable or `MLFLOW_ENABLED=false`, `get_tracer()` falls back to a `NoopTracer` so the chat is unaffected. Per-node latency is *not* captured (only whole-graph `graph_invoke`).

## Architecture

A document QA system: a **LangGraph** agent answers questions grounded in ingested documents, using hybrid retrieval + multi-query fusion + **Qwen 3.8 27B** generation (via **Groq**'s free tier by default; Hugging Face Inference Providers optional). Embeddings are **`qwen/qwen3-embedding-8b` via OpenRouter**. Gemini is used only for image captions. There is **no reranker** and no torch in the default install (local embeddings are optional via `requirements-local.txt`). Two entry points share the same core (`ingestion/`, `vectorstore/`, `agents/`, `monitoring/`):

- **`webapp.py`** — the FastHTML web UI (per-session collections, MLflow tracing).
- **`chat.py`** — a CLI that ingests a doc and loops Q&A through the same graph.

Both run in-process: embedded Qdrant + synchronous ingest + local SQLite MLflow, no Docker or local models required.

### Agent graph ([agents/rag_agent.py](agents/rag_agent.py))

`query_analyzer → retriever → (context_builder → generator) → END`, with a retry loop back into `query_analyzer` (on empty retrieval) or `retriever` (on insufficient). Each node is a pure `run(state) -> dict` function in [agents/nodes/](agents/nodes/) returning a partial `AgentState` update. State shape is the single source of truth in [agents/state.py](agents/state.py).

**Retry loop subtlety:** the loop is bounded by `iteration_count < 3` in `_route_after_retrieval`, but **both** `query_analyzer` and `retriever` increment `iteration_count` by 1, so a full empty-retry cycle advances it by 2. The retriever's own increment is what actually prevents the infinite loop — don't remove it. Retrieval quality is classified purely by chunk count: 0 → `empty`, <3 → `insufficient`, else `sufficient`.

**Context selection:** the retriever runs one hybrid search per expanded sub-query and **sums each chunk's RRF score across sub-queries**, sorts, and returns `raw_chunks` (all) + `top_chunks` (`CONTEXT_TOP_N`, default 6). `context_builder` consumes `top_chunks`.

### Vector store ([vectorstore/](vectorstore/))

Hybrid search = dense (`all-mpnet-base-v2`, 768-dim, cosine) + BM25 sparse, fused server-side via Qdrant RRF (`Prefetch` + `FusionQuery`, not dicts). Collections are auto-created lazily by `ensure_collection`.

Several invariants here are load-bearing and were the source of past bugs — preserve them:
- **One shared `QdrantClient` per process** (`_get_client` in [vectorstore/qdrant_store.py](vectorstore/qdrant_store.py)). In-process mode keeps data inside a single client instance and takes an exclusive file lock on `qdrant_local/`; ingest and retriever must use the same client, and two clients cannot open the path at once.
- **Chunk IDs are `uuid5`** of `filename-page_num-sheet_name-row_range-<full content>` ([semantic_chunker.py](ingestion/chunkers/semantic_chunker.py) `_make_id`) — Qdrant point IDs must be UUID/uint, and uuid5 keeps identical content stable/idempotent. Never key on a content *prefix*: it used to be `content[:64]`, which made spreadsheet row-windows (same header) collide and silently overwrite each other.
- **BM25 buckets use `hashlib.blake2b`, not builtin `hash()`** ([vectorstore/sparse_index.py](vectorstore/sparse_index.py)) — builtin hash is salted per process, so ingest-time and query-time vectors wouldn't match. Token buckets are aggregated into a dict because Qdrant rejects sparse vectors with duplicate indices.

### LLM client ([agents/llm.py](agents/llm.py))

Query analysis and generation go through `agents/llm.py` (`chat`, `chat_stream`), a lazily-initialized `huggingface_hub.InferenceClient`. `LLM_PROVIDER=groq` (default) points it at `https://api.groq.com/openai/v1` with `GROQ_API_KEY` and model `qwen/qwen3.8-27b`; `LLM_PROVIDER=hf` uses HF Inference Providers (`HF_TOKEN`, `HF_PROVIDER`, `Qwen/Qwen3.8-27B`). `LLM_MODEL` overrides the id. Qwen3 can emit `<think>…</think>`: we send the provider's no-thinking flag (Groq `reasoning_effort=none`, HF `chat_template_kwargs.enable_thinking=False`) **and** strip think blocks (`strip_think`, and `filter_think` for streams) — keep both, provider support varies. JSON from `query_analyzer` is parsed with `llm.parse_json` (fence/prose tolerant, `{}` on failure, then type-checked defaults); we don't rely on `response_format`.

Groq's free tier is small (~8K tokens/min, ~200K/day), so prompts are capped: `MAX_ANSWER_TOKENS` (1024) and `MAX_CONTEXT_CHARS` (16000).

**Embeddings** (`get_embedder()` in `ingestion/embedders/__init__.py`, providers imported lazily): `EMBEDDING_PROVIDER=openrouter` (default, [openrouter_embedder.py](ingestion/embedders/openrouter_embedder.py): `OPENROUTER_API_KEY`, batches of 64 sent `EMBED_CONCURRENCY` (4) at a time, retries 429/5xx, query-side instruction prefix since Qwen3-Embedding is asymmetric, 4096-dim output truncated to `EMBEDDING_DIM`=768 + renormalized — valid because the model is Matryoshka-trained), `gemini`, or `local` (needs `requirements-local.txt`). Changing provider/dim invalidates stored vectors; web sessions are ephemeral so that only matters for `chat.py`'s `default` collection.

Gemini (`google-genai`) remains in `image_loader` (`VISION_MODEL`) and the optional Gemini embedder; lazy via `_get_client()` because `genai.Client` validates `GOOGLE_API_KEY` at construction. Keep the lazy pattern for any new API client so unit tests run with no keys. A failed caption is non-fatal if OCR found text.

### Ingestion ([ingestion/pipeline.py](ingestion/pipeline.py))

`load → chunk → embed → upsert`, dispatched by file extension (PDF / spreadsheet / image / text). **Each `_ingest_*` collects all of a file's chunks and calls `_embed_and_upsert` once.** Don't go back to per-page calls: each embedding request costs about 1 s of round trip, and per-page embedding took a 138-page PDF from about 12 s to about 8 min. PDF tables pass through `pdf_loader.is_real_table`, which drops the slide layout boxes PyMuPDF reports as tables. PDF images are not extracted. PDFs split text and tables separately; spreadsheets/DB tables emit a schema chunk plus 20-row windows. The DB loader strips credentials from the stored `filename`.

### Deployment (Hugging Face Spaces)

`Dockerfile` + README front-matter (`sdk: docker`, `app_port: 7860`) make the repo a Docker Space: `uvicorn webapp:app --port 7860 --workers 1` (one worker — embedded Qdrant lock + per-process counters), runs as uid 1000, `MLFLOW_ENABLED=false`, ephemeral storage. Secrets: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY` (image captions), `SESSION_SECRET`.

## Configuration

All settings are Pydantic `BaseSettings` in [config/settings.py](config/settings.py), loaded from `.env`. Key knobs: `GROQ_API_KEY` + `OPENROUTER_API_KEY` (required), `GOOGLE_API_KEY` (image captions), `LLM_PROVIDER` (`groq`), `EMBEDDING_PROVIDER` (`openrouter`), `VISION_MODEL` (`gemini-2.5-flash`), `RETRIEVER_TOP_K` (20), `CONTEXT_TOP_N` (6), and `QDRANT_LOCAL_PATH` — set to a folder path for in-process Qdrant (`chat.py` sets this before importing any project module, which is required for the override to take effect) or `:memory:` for ephemeral.

## Gotchas

- `webapp.py` must run with `reload=False` (already set): the auto-reloader spawns a second process that collides with the embedded Qdrant file lock (`AlreadyLocked`). FastHTML `@rt(path)` registers GET+POST by default. If port 5001 won't rebind after a kill, an orphaned `multiprocessing.spawn` child may still hold the socket — find it via `Get-CimInstance Win32_Process` and `taskkill /F`.
- Providers history: Gemini (free-tier limits) → Qwen via HF Inference Providers (free credits ran out, 402) → Qwen via **Groq** + **OpenRouter** embeddings. `Settings` uses `extra="ignore"`, so stale keys in an old `.env` (e.g. `GENERATION_MODEL`, `RERANKER_MODEL`) are harmless.
- Public demo guard in `webapp.py` `/ask`: `MAX_QUESTIONS_PER_SESSION` (count in the signed cookie, survives `/reset`) and `DAILY_QUESTION_LIMIT` (process-wide). Both 0 = off.

## Guardrails
- Make sure you DO NOT answer question if the user query is not related to the source document.
