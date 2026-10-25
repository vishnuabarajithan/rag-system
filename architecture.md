# RAG System — Architecture

How the document Q&A app works, end to end: which module does what, what runs
locally versus as a hosted API, how the vector database is set up, and how a
document and a question move through the system.

> Kept in sync with the code as of October 2026. Where this file and the code
> disagree, trust the code, starting with [`config/settings.py`](config/settings.py).

---

## Contents

1. [Overview](#1-overview)
2. [System diagram](#2-system-diagram)
3. [What runs locally and what calls an API](#3-what-runs-locally-and-what-calls-an-api)
4. [Repository map](#4-repository-map)
5. [Ingestion flow: upload to stored vectors](#5-ingestion-flow-upload-to-stored-vectors)
6. [The vector database (Qdrant)](#6-the-vector-database-qdrant)
7. [Question flow: question to streamed answer](#7-question-flow-question-to-streamed-answer)
8. [The agent graph (LangGraph)](#8-the-agent-graph-langgraph)
9. [Web UI and sessions](#9-web-ui-and-sessions)
10. [Grounding and guardrails](#10-grounding-and-guardrails)
11. [Configuration reference](#11-configuration-reference)
12. [Monitoring (MLflow)](#12-monitoring-mlflow)
13. [Deployment](#13-deployment)
14. [Failure modes and limits](#14-failure-modes-and-limits)
15. [Testing](#15-testing)

---

## 1. Overview

Users upload documents (PDF, TXT, MD, CSV, XLSX/XLS, images) and ask questions
about them. Every answer is grounded in the uploaded content and cites its
sources with numbered `[N]` markers. Clicking a marker shows the exact passage it
came from.

The pipeline has three parts:

- **Ingestion.** Files are parsed, split into chunks, and each chunk gets two
  representations: a dense semantic vector (from an embedding API) and a sparse
  BM25-style keyword vector (computed locally). Both are stored in Qdrant.
- **Retrieval.** A LangGraph agent rewrites the question into 1–3 sub-queries.
  Each sub-query runs a hybrid (dense + keyword) search, and the results are
  fused into one ranked list.
- **Generation.** The top chunks become a numbered context, and the LLM streams
  an answer that may only use that context.

There are two entry points that share the same core code:

| Entry point | What it is | Storage |
|---|---|---|
| [`webapp.py`](webapp.py) | FastHTML web chat UI on http://localhost:5001 (port 7860 in Docker). One isolated, short-lived session per browser. | One `session_<sid>` collection per session, discarded on refresh |
| [`chat.py`](chat.py) | Command-line tool: ingest a file, then ask questions in a loop. | The `default` collection, which persists in `./qdrant_local/` |

Everything except the model calls runs **in one Python process**: the web
server, the vector database (embedded Qdrant), parsing, chunking, keyword
encoding, the agent graph, the session store and monitoring. No Docker, no
database server and no local GPU models are needed.

---

## 2. System diagram

```
                                  ┌──────────────────────────── EXTERNAL APIs (HTTPS) ───────────────────────────┐
                                  │                                                                              │
                                  │   Groq API                  OpenRouter API              Google Gemini API    │
                                  │   qwen/qwen3.8-27b          qwen/qwen3-embedding-8b     gemini-2.5-flash     │
                                  │   (query analysis +         (dense embeddings,          (image captions      │
                                  │    answer streaming)         4096 → 768 dims)            only)               │
                                  └──────▲────────────────────────────▲──────────────────────────▲───────────────┘
                                         │ chat completions           │ /embeddings              │ generate_content
                                         │                            │                          │
┌──────────┐  HTTP + SSE   ┌─────────────┴────────────────────────────┴──────────────────────────┴───────────────┐
│ Browser  │◄─────────────►│  ONE PYTHON PROCESS (uvicorn / FastHTML)                                            │
│ (HTMX,   │  signed       │                                                                                     │
│  <dialog>│  cookie: sid  │  webapp.py ── routes: /  /upload  /toggle  /delete  /ask  /stream  /reset           │
│  preview)│               │     │                                                                               │
└──────────┘               │     ├── web/sessions.py ............ per-session state (memory or Redis)           │
                           │     │                                                                               │
                           │     ├── ingestion/pipeline.py ...... load → chunk → embed → BM25 → upsert          │
                           │     │     loaders/ (PyMuPDF, pandas, Tesseract OCR*)                                │
                           │     │     chunkers/semantic_chunker.py                                              │
                           │     │     embedders/ (OpenRouter | Gemini | local)                                  │
                           │     │                                                                               │
                           │     ├── agents/rag_agent.py ........ LangGraph: query_analyzer → retriever          │
                           │     │     agents/llm.py (LLM client)        ↺ retry → context_builder → generator   │
                           │     │                                                                               │
                           │     ├── vectorstore/ ............... QdrantStore, collection schema, BM25 encoder   │
                           │     │        │                                                                      │
                           │     │        ▼                                                                      │
                           │     │   Embedded Qdrant (in-process, ./qdrant_local/, exclusive file lock)          │
                           │     │                                                                               │
                           │     └── monitoring/mlflow_tracer.py  → ./mlflow.db (SQLite)                         │
                           └─────────────────────────────────────────────────────────────────────────────────────┘
                                                                                * if the Tesseract binary is installed
```

---

## 3. What runs locally and what calls an API

### Local (inside the Python process)

| Component | Library / module | Notes |
|---|---|---|
| Web server and UI | FastHTML (Starlette + Uvicorn), HTMX, HTMX SSE extension | [`webapp.py`](webapp.py). Pages are rendered on the server; answers stream over Server-Sent Events. |
| Vector database | `qdrant-client` in **embedded mode** | Data lives in `./qdrant_local/`. The process holds an exclusive file lock, so only one process can open it. `:memory:` mode is used in tests. |
| Keyword (sparse) vectors | [`vectorstore/sparse_index.py`](vectorstore/sparse_index.py) | Hash-bucketed BM25-style term weights. Pure Python, no model. |
| PDF parsing | PyMuPDF (`fitz`) | Page text, tables (`find_tables`), and page rendering for OCR. |
| Spreadsheet parsing | pandas + openpyxl (`.xlsx`) / xlrd (`.xls`) | CSV through `pd.read_csv`. |
| OCR | `pytesseract` + the **Tesseract binary** | Optional. If the binary is missing, OCR quietly returns nothing. Installed in the Docker image; **not installed on the dev PC**. |
| Chunking | `langchain-text-splitters` `RecursiveCharacterTextSplitter` | 512 characters, 64 overlap. |
| Agent orchestration | LangGraph | [`agents/rag_agent.py`](agents/rag_agent.py) |
| Session store | In-memory dict (default) or Redis | [`web/sessions.py`](web/sessions.py) |
| Monitoring | MLflow with a SQLite tracking store | [`monitoring/mlflow_tracer.py`](monitoring/mlflow_tracer.py) → `./mlflow.db` |
| Markdown rendering | `markdown-it-py` | Raw HTML in model output stays escaped. |

### External APIs

| Purpose | Provider and model | Client | Key | Cost and limits |
|---|---|---|---|---|
| Query analysis + answer generation | **Groq**, `qwen/qwen3.8-27b` (default) | `huggingface_hub.InferenceClient` pointed at `https://api.groq.com/openai/v1` ([`agents/llm.py`](agents/llm.py)) | `GROQ_API_KEY` | Free tier, roughly 30 requests/min, 8K tokens/min, 1K requests/day and 200K tokens/day. Groq lists this model as a *preview*, so it can be withdrawn at short notice. |
| Same, alternative | **Hugging Face Inference Providers**, `Qwen/Qwen3.8-27B` (`LLM_PROVIDER=hf`) | `InferenceClient(provider=HF_PROVIDER)` | `HF_TOKEN` | Paid credits. The free monthly allowance on this account is used up (calls return 402). |
| Dense embeddings | **OpenRouter**, `qwen/qwen3-embedding-8b` (default) | `httpx` POST to `https://openrouter.ai/api/v1/embeddings` ([`openrouter_embedder.py`](ingestion/embedders/openrouter_embedder.py)) | `OPENROUTER_API_KEY` | About $0.01 per million tokens. Needs a small prepaid balance. |
| Dense embeddings, alternative | Google `gemini-embedding-001` (`EMBEDDING_PROVIDER=gemini`) | `google-genai` | `GOOGLE_API_KEY` | Free tier allows 100 requests/min. That limit caused the original upload failures. |
| Dense embeddings, alternative | Local `all-mpnet-base-v2` (`EMBEDDING_PROVIDER=local`) | sentence-transformers on CPU | none | Free and unlimited, but needs `requirements-local.txt` (torch). |
| Image descriptions | **Google Gemini** `gemini-2.5-flash` (`VISION_MODEL`) | `google-genai` ([`image_loader.py`](ingestion/loaders/image_loader.py)) | `GOOGLE_API_KEY` | Free tier. Thinking is turned off so the caption isn't cut short. If it fails, the upload still succeeds as long as OCR found text. |

All API clients are created **lazily**, on first use, through a module-level
`_get_client()`. Importing a module never needs a key, so unit tests run
without any credentials.

---

## 4. Repository map

```
rag_system/
├── webapp.py                  Web UI: routes, HTML components, CSS, citation dialog, demo caps, sessions
├── chat.py                    CLI: ingest one file into `default`, then a Q&A loop over rag_graph
├── config/
│   └── settings.py            Every setting (Pydantic BaseSettings, loaded from .env, extra="ignore")
├── agents/
│   ├── llm.py                 LLM client: provider switch (groq/hf), chat(), chat_stream(), think-stripping, parse_json()
│   ├── rag_agent.py           LangGraph graphs: rag_graph (full) and context_graph (stops before generation)
│   ├── state.py               AgentState / RetrievedChunk TypedDicts, the single source of truth for graph state
│   └── nodes/
│       ├── query_analyzer.py  LLM → JSON: sub-queries, intent, filters
│       ├── retriever.py       Embed sub-queries → hybrid search → cross-query score fusion → top_chunks
│       ├── context_builder.py Numbered [N] context blocks + citation records (with excerpts)
│       └── generator.py       Grounded answer: run() (blocking) and stream_generate() (streaming)
├── ingestion/
│   ├── pipeline.py            IngestionPipeline: dispatch by extension → chunk → embed → BM25 → upsert
│   ├── loaders/
│   │   ├── pdf_loader.py      Page text (OCR fallback for scanned pages), tables as markdown, embedded images
│   │   ├── text_loader.py     Plain text / markdown
│   │   ├── spreadsheet_loader.py  XLSX/XLS/ODS (every sheet) and CSV (one "sheet")
│   │   ├── image_loader.py    Tesseract OCR + Gemini caption → one combined text
│   │   └── database_loader.py SQLAlchemy table extraction (used by pipeline.ingest_database; not wired into the UI)
│   ├── chunkers/
│   │   └── semantic_chunker.py  ChunkingPipeline: text splits, table schema + 20-row windows, image chunk, uuid5 IDs
│   └── embedders/
│       ├── __init__.py        get_embedder(): shared singleton, picks the provider, imports it lazily
│       ├── openrouter_embedder.py  Default. Batching, retry, query instruction, truncate + renormalize
│       ├── gemini_embedder.py      Alternative (Gemini API)
│       ├── text_embedder.py        Alternative (local sentence-transformers)
│       └── image_embedder.py       Unused (CLIP-style, imports torch)
├── vectorstore/
│   ├── qdrant_store.py        One shared QdrantClient per process; QdrantStore: hybrid_search, upsert, delete, drop
│   ├── collections.py         ensure_collection(): lazy creation of named dense + sparse vectors and payload indexes
│   └── sparse_index.py        bm25_encode(): blake2b hash buckets → (indices, values)
├── web/
│   └── sessions.py            InMemorySessionStore / RedisSessionStore, collection_for(sid) = "session_<sid>"
├── monitoring/
│   └── mlflow_tracer.py       RAGTracer / NoopTracer, get_tracer(): one MLflow run per question, crash-safe
├── tests/
│   ├── unit/                  No network or keys needed
│   └── integration/           Live LLM call (needs GROQ_API_KEY)
├── Dockerfile, .dockerignore  Hugging Face Spaces image (no torch, includes Tesseract)
├── requirements.txt           Base dependencies (no torch)
├── requirements-local.txt     Optional torch / sentence-transformers for EMBEDDING_PROVIDER=local
├── .env / .env.example        Keys and settings (.env is gitignored)
├── qdrant_local/              Embedded Qdrant data (gitignored)
├── mlflow.db, mlruns/         MLflow tracking store and artifacts (gitignored)
└── api/routes/                Empty leftover folder from an earlier REST API; unused
```

Settings that exist but **have no effect**: `cache_similarity_threshold`,
`cache_ttl_seconds` (the Redis session store reads it but never applies it as
an expiry), `api_host`, `api_port` and `api_workers`, all left over from
earlier designs. `qdrant_collection`
(`rag_documents`) is only the default argument of `ensure_collection`; real
calls always pass a collection name.

---

## 5. Ingestion flow: upload to stored vectors

### 5.1 Sequence

```
Browser                webapp.py /upload            IngestionPipeline              OpenRouter        Qdrant (embedded)
   │ multipart POST           │                            │                            │                  │
   │ (one or more files) ───► │                            │                            │                  │
   │                          │ for each file:             │                            │                  │
   │                          │  skip if duplicate/empty   │                            │                  │
   │                          │  ingest_file(name, bytes,  │                            │                  │
   │                          │     "session_<sid>") ────► │ write temp file            │                  │
   │                          │                            │ dispatch by extension      │                  │
   │                          │                            │ loader → chunker           │                  │
   │                          │                            │ embed_batch(chunk texts) ─►│ (batches of 64)  │
   │                          │                            │◄─────── 4096-d vectors ────│                  │
   │                          │                            │ truncate→768, normalize    │                  │
   │                          │                            │ bm25_encode(each chunk)    │                  │
   │                          │                            │ upsert(points) ─────────────────────────────► │
   │                          │◄──── chunk count ──────────│ delete temp file           │                  │
   │                          │  0 chunks → "no readable text" error for that file                        │
   │◄──── docs panel (HTML) ──│  failures listed; successes appear checked in the list                    │
```

### 5.2 Step by step

1. **Upload route** ([`webapp.py` `upload`](webapp.py)). Takes a list of files.
   Each file is handled **independently**. A duplicate filename (within the
   session) or an empty file is skipped. An exception or a result of 0 chunks
   is collected as a per-file error ("Couldn't index X: …"), and the other
   files still go through. Successful files are added to `state["docs"]` and
   switched on in `state["enabled"]`.
2. **Dispatch** ([`IngestionPipeline.ingest_file`](ingestion/pipeline.py)). The
   bytes are written to `%TEMP%/<uuid><ext>` and routed by extension:

   | Extension | Path |
   |---|---|
   | `.pdf` | `_ingest_pdf` |
   | `.xlsx .xls .ods .xlsm .csv` | `_ingest_spreadsheet` |
   | `.png .jpg .jpeg .gif .webp .bmp .tiff` | `_ingest_image` |
   | anything else (`.txt .md .rst .log` …) | `_ingest_text` |

   The temp file is always deleted afterwards. That's why loaders must close
   their file handles: on Windows an open handle makes the delete fail
   (`WinError 32`), which is why the Excel reader uses a `with` block.

3. **Loading** (one loader per format):
   - **PDF** ([`pdf_loader.py`](ingestion/loaders/pdf_loader.py)): for each page,
     `page.get_text()`. If a page has **no text layer** (a scanned page), it's
     rendered at 200 DPI and OCR'd with Tesseract; without Tesseract the page
     comes out empty. Tables come from `page.find_tables()` and go through
     `is_real_table`, which drops the layout boxes the detector reports on
     slides. A table is dropped if it's smaller than 2×2, if more than half its
     cells are empty or auto-named `ColN`, or if it has 2 columns and ≥ 90% of
     its words already appear in the page text. (On a 138-slide lecture deck
     this removed all 257 false "tables".) Kept tables become markdown. Embedded
     images are **not** extracted or indexed, so text that exists only inside a
     picture (a diagram, or a table pasted as an image) can't be retrieved.
   - **TXT/MD** ([`text_loader.py`](ingestion/loaders/text_loader.py)): the whole
     file as UTF-8 (bad bytes replaced) is a single "page".
   - **XLSX/XLS/ODS** ([`spreadsheet_loader.py`](ingestion/loaders/spreadsheet_loader.py)):
     every sheet becomes a DataFrame. Empty rows are dropped and NaN becomes `""`.
   - **CSV**: one DataFrame. Its "sheet name" is the *original* file stem (the
     pipeline passes `name=`, because the temp file has a random name).
   - **Images** ([`image_loader.py`](ingestion/loaders/image_loader.py)): Tesseract
     OCR text **plus** a Gemini description (prompted to transcribe all text,
     numbers and labels; thinking disabled; up to 1,024 tokens). If the caption
     fails, OCR alone is enough. If both come back empty, the upload fails with a
     clear message.

4. **Chunking** ([`semantic_chunker.py`](ingestion/chunkers/semantic_chunker.py)):
   - **Text** (PDF pages, TXT, MD): `RecursiveCharacterTextSplitter`, 512
     characters with 64 overlap, splitting on paragraph, then line, then
     sentence, then word. `content_type="text"`.
   - **Tables** (spreadsheets and CSV): one **schema chunk** (`Table: <sheet> (from
     <file>)`, the column names and the row count; `is_schema=True`), then one
     chunk per **20-row window**. Each window starts with
     `Table: <sheet> (from <file>), rows 1-20` so a question naming the sheet or
     file can match it, followed by the rows as a markdown table.
     `content_type="table"`, with metadata `row_range`.
   - **PDF tables**: a single markdown chunk. There's no schema chunk because
     the columns aren't known.
   - **Images**: one chunk holding `[OCR TEXT]: …` + `[VISUAL DESCRIPTION]: …`,
     `content_type="image_caption"`.
   - **Chunk IDs**: `uuid5(NAMESPACE_URL, "filename-page_num-sheet_name-row_range-<full text>")`.
     Qdrant point IDs must be a UUID or an unsigned integer, and uuid5 is
     deterministic, so re-ingesting the same content overwrites rather than
     duplicates. The key uses the **full** text on purpose: it used to be the
     first 64 characters, so every 20-row window of a sheet (all starting with
     the same table header) got the same ID and overwrote the one before, and
     only the last rows of each sheet survived.

5. **Dense embedding** ([`openrouter_embedder.py`](ingestion/embedders/openrouter_embedder.py)):
   each `_ingest_*` method first collects **every chunk of the file**, then
   calls `_embed_and_upsert` **once**. `embed_batch(texts)` splits the texts
   into requests of 64 and sends up to `EMBED_CONCURRENCY` (4) of them at the
   same time through a thread pool; `map()` keeps the results in input order.
   This matters because each request has about 1 s of fixed round-trip cost
   whatever its size: embedding page by page meant 395 sequential requests and
   about 8 minutes for a 138-slide deck, against 3 parallel requests and about
   12 s now. The total token cost is the same either way.
   Documents are sent **without** the query instruction (Qwen3-Embedding treats
   queries and documents differently; see §7). The 4096-dim output is
   **truncated to `EMBEDDING_DIM` = 768 and renormalized** to unit length. This
   is valid because the model is trained so its leading dimensions carry the
   most meaning (Matryoshka training). It keeps the Qdrant schema at 768 no
   matter which provider is used. On a 429 or 5xx response it retries up to 4
   times, waiting `Retry-After` or 1/2/4/8 s. Any other error, or an error body
   returned with HTTP 200, raises with the provider's message.

6. **Sparse (keyword) encoding** ([`sparse_index.py`](vectorstore/sparse_index.py)):
   lowercased `\w+` tokens, each hashed with **`blake2b`** into one of 30,000
   buckets. Python's built-in `hash()` would be wrong here: it's randomized per
   process, so vectors made at ingest time wouldn't match those made at query
   time. Each bucket's weight is the BM25 term-frequency part
   (`k1=1.5`, `b=0.75`, assumed average length 100). There's no corpus-wide IDF.
   Tokens that land in the same bucket are **added together**, because Qdrant
   rejects sparse vectors with duplicate indices.

7. **Upsert** ([`QdrantStore.upsert`](vectorstore/qdrant_store.py)): creates the
   collection if needed (`ensure_collection`), then writes `PointStruct(id, vector={"dense": [...], "sparse": SparseVector(...)}, payload)`,
   in slices of 256 points per write (`_UPSERT_BATCH`).
   The payload includes `content`, `content_type`, `document_id`, `filename`,
   `file_type`, `page_num`, `sheet_name`, `row_range`, `row_count`,
   `is_schema` and `created_at`, depending on the source.

Ingestion runs **synchronously** inside the request. While it runs, the upload
button is disabled and reads "Indexing…": HTMX adds `.htmx-request` to the
form, and CSS swaps the button label.

---

## 6. The vector database (Qdrant)

### 6.1 Deployment modes ([`_get_client`](vectorstore/qdrant_store.py))

| `QDRANT_LOCAL_PATH` | Mode | Used by |
|---|---|---|
| a folder path | **Embedded**, persisted on disk | `webapp.py` and `chat.py` (both default to `./qdrant_local` through `os.environ.setdefault` **before** importing settings), and the Docker image (`/home/user/app/qdrant_local`) |
| `:memory:` | Embedded, lost on exit | Tests |
| empty | **Server mode**: `QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)` | Scaled-out production |

**One shared client per process.** In embedded mode the data lives inside a
single client instance, which holds an exclusive lock on the folder. The
ingestion pipeline, the retriever and the web routes all construct
`QdrantStore()`, but they share one module-level `QdrantClient`. A second
process (a second `webapp.py`, or `chat.py` running alongside the web app)
fails with `AlreadyLocked`. This is also why `webapp.py` runs with
`reload=False`: the auto-reloader would start a second process.

### 6.2 Collection schema ([`ensure_collection`](vectorstore/collections.py))

Collections are created **on demand**, the first time anything reads from or
writes to them.

| Part | Definition |
|---|---|
| Named dense vector `dense` | 768 dims (`EMBEDDING_DIM`), **cosine** distance |
| Named sparse vector `sparse` | Sparse index held in memory (`on_disk=False`) |
| Payload indexes | `filename`, `file_type`, `content_type` (keyword); `page_num` (integer); `created_at` (datetime) |

Changing the embedding provider or dimension makes stored vectors
incompatible. Web sessions are short-lived, so that only matters for `chat.py`'s
persistent `default` collection, which still holds vectors from older
providers. To clear it, delete `qdrant_local/` while the app is stopped.

### 6.3 Hybrid search ([`QdrantStore.hybrid_search`](vectorstore/qdrant_store.py))

A single `query_points` call:

```
prefetch = [
  Prefetch(query=<dense vector>,  using="dense",  limit=2*top_k, filter=F),
  Prefetch(query=<sparse vector>, using="sparse", limit=2*top_k, filter=F),
]
query = FusionQuery(fusion=Fusion.RRF)      # Reciprocal Rank Fusion, done inside Qdrant
limit = top_k                               # RETRIEVER_TOP_K = 20
```

- **F** combines (AND) any `query_filters` the query analyser extracted
  (exact-match `FieldCondition`s) with
  `FieldCondition(key="filename", match=MatchAny(any=<checked sources>))`, which
  is how the sidebar checkboxes limit retrieval to the selected files.
- RRF combines the two ranked lists by rank, not raw score, so the dense and
  keyword legs don't need comparable scales. The resulting `score` is the RRF
  score.

### 6.4 Isolation and cleanup

- Each browser session gets its own collection, **`session_<sid>`**
  ([`collection_for`](web/sessions.py)). Sessions can't see each other's data
  because retrieval only ever queries the session's own collection.
- **Deleting a document** removes its points with a `filename` filter
  (`delete_by_filename`).
- **Refresh or Reset** drops the whole collection (`_discard_session`).
- **At startup**, in embedded mode only, every collection whose name starts with
  `session_` is dropped (`drop_collections_with_prefix`). These belong to
  sessions from a previous run that can no longer be reached. In server mode
  this is **skipped**, because another worker's live sessions would be deleted.
- Collections from tabs that are simply closed stay until the next server start.

---

## 7. Question flow: question to streamed answer

### 7.1 Sequence

```
Browser              /ask                 /stream (SSE)          context_graph (LangGraph)                 Groq        OpenRouter   Qdrant
  │ POST question ──► │                      │                          │                                    │             │           │
  │                   │ any source checked?  │                          │                                    │             │           │
  │                   │ demo caps OK?        │                          │                                    │             │           │
  │                   │ q_count += 1         │                          │                                    │             │           │
  │                   │ state.pending = {…}  │                          │                                    │             │           │
  │◄ user bubble +    │                      │                          │                                    │             │           │
  │  SSE placeholder ─┘                      │                          │                                    │             │           │
  │ GET /stream ────────────────────────────►│ invoke ─────────────────►│ query_analyzer ───────────────────►│ JSON        │           │
  │                                          │                          │ retriever: per sub-query ──────────────────────► │ embed     │
  │                                          │                          │            hybrid_search ─────────────────────────────────────►│
  │                                          │                          │  ↺ retry if empty/insufficient                                 │
  │                                          │                          │ context_builder ([N] blocks, citations)                        │
  │                                          │◄─── ctx ─────────────────┘                                    │             │           │
  │                                          │ stream_generate ────────────────────────────────────────────►│ stream      │           │
  │◄── event: token (markdown so far) ×N ────│◄────────────────────────── deltas (thinking stripped) ────────│             │           │
  │◄── event: done (final bubble) ───────────│ save message, log to MLflow                                   │             │           │
```

### 7.2 `/ask` ([`webapp.py` `ask`](webapp.py))

1. Adds the user's message to the session's chat.
2. If **no source is checked**, replies "Select at least one source…" with no LLM call.
3. **Demo caps** (`_limit_reason`):
   - per visitor: `session["q_count"] >= MAX_QUESTIONS_PER_SESSION` (20). The
     count lives in the **signed cookie**, so starting a new session (refresh
     or Reset only changes `sid`) doesn't reset it.
   - per process per day: a module-level `{date, count}` checked against
     `DAILY_QUESTION_LIMIT` (80, sized for Groq's ~200K tokens/day).
   - when a cap is hit: a friendly message, and no LLM call.
4. Increments the counters, stores `state["pending"] = {question, selected, mid}`,
   and returns right away: the user's bubble plus a placeholder `stream_shell`
   that opens an SSE connection to `/stream`. That's why the question appears
   immediately, before any model work.

### 7.3 `/stream` ([`webapp.py` `stream`](webapp.py))

Runs `context_graph.invoke({...})` (retrieval only, §8), then streams the answer.

**a) `query_analyzer`** ([`query_analyzer.py`](agents/nodes/query_analyzer.py)), an LLM call:
- The system prompt asks for JSON only, with `expanded_queries` (1–3 different
  rephrasings), `query_intent` (`factual | analytical | summarize | compare`),
  `query_filters` (metadata filters stated explicitly in the question),
  `requires_table_reasoning` and `requires_image_understanding`.
- On a retry pass (`iteration_count > 0`) it's told the last search found
  nothing and asked for broader sub-queries.
- Output goes through [`llm.parse_json`](agents/llm.py): strips `<think>`
  blocks and code fences, takes the first `{…}`, returns `{}` if that fails.
  Each field is then type-checked (a non-list `expanded_queries` falls back to
  the original question; a non-dict `query_filters` falls back to `{}`).
- `max_tokens=512`. Adds 1 to `iteration_count`.

**b) `retriever`** ([`retriever.py`](agents/nodes/retriever.py)), no LLM:
- For each sub-query: `embed(query)` sends `"Instruct: Given a question, retrieve
  passages from the user's documents that answer it\nQuery: <q>"` to OpenRouter.
  This query-side instruction is how Qwen3-Embedding is meant to be used.
  Then `hybrid_search(top_k=20, filters, filenames=<checked sources>)`.
- **Fusion across sub-queries**: results are merged by `chunk_id` and **their
  RRF scores are added**, so chunks several sub-queries agree on rise to the top.
  The merged list is sorted by score.
- Returns `raw_chunks` (everything, sorted), `top_chunks` (the first
  `CONTEXT_TOP_N` = 6) and `retrieval_quality` from the **total** count: 0 is
  `empty`, fewer than 3 is `insufficient`, otherwise `sufficient`. Adds 1 to
  `iteration_count`.
- There's **no reranker**. It was removed to keep torch and model downloads out
  of the deployment; cross-query fusion does the final ordering.

**c) Retry routing** (`_route_after_retrieval` in [`rag_agent.py`](agents/rag_agent.py)):

| Condition | Next node |
|---|---|
| `empty` and `iteration_count < 3` | `query_analyzer` (widen the sub-queries) |
| `insufficient` and `iteration_count < 3` | `retriever` (search again) |
| `error` set | END |
| otherwise | `context_builder` |

Both `query_analyzer` and `retriever` add 1 to `iteration_count`, so an
"empty" retry cycle uses 2. The retriever's own increment is what stops an
infinite loop on the `insufficient → retriever` edge; **don't remove it**.
After the cap, the graph moves on with whatever it has (possibly nothing), and
the model then says the documents don't cover the question.

**d) `context_builder`** ([`context_builder.py`](agents/nodes/context_builder.py)), no LLM:
- Each top chunk becomes a numbered block:
  - text: `[N] [Source: <file>, p.<page>]` + the content
  - table: `[N] [TABLE from <file>, sheet=<sheet>, rows=<range>]` + the markdown
  - image: `[N] [IMAGE from <file>]` + OCR text and caption
- Stops adding blocks at `MAX_CONTEXT_CHARS` = 16,000 characters (about 4K
  tokens) to stay inside Groq's per-minute token limit.
- Creates a **citation record** per block: `{ref "[N]", chunk_id, source_file,
  page_num, sheet_name, content_type, content: first 1,500 characters}`. The
  excerpt is what the preview dialog shows.

**e) Generation** ([`generator.stream_generate`](agents/nodes/generator.py) → [`llm.chat_stream`](agents/llm.py)):
- System prompt: answer **only** from the context, cite with `[N]`, say so
  clearly if the context isn't enough, never invent anything, show the working
  for table calculations. The user message is the context, the question, and a
  hint for the detected intent (e.g. "compare" asks for a table or bullets).
- `max_tokens = MAX_ANSWER_TOKENS` (1,024). Streaming is on, with
  `stream_options.include_usage` for token counts. Groq sends usage in either
  `usage` or `x_groq.usage`, and both are handled.
- **Turning Qwen's thinking off**: the request sets Groq's
  `reasoning_effort: "none"` (or HF's `chat_template_kwargs.enable_thinking=false`).
  As a second line of defence, `filter_think` removes any `<think>…</think>`
  block that still appears at the start of the stream, even when the tag is
  split across chunks. Both are kept on purpose, because provider support
  varies.
- Each piece of streamed text causes the **whole answer so far** to be
  re-rendered as markdown and sent as an SSE `token` event, which HTMX swaps
  into the placeholder.

**f) Finish**: confidence = `min(1, len(top_chunks) / CONTEXT_TOP_N)`, which
measures how full the context was, not how accurate the answer is. The message
is saved to the session. MLflow logs the run (§12). A final `done` event
replaces the placeholder with the finished bubble: `[N]` links, cited source
chips and a metadata line (intent, retrieval quality, confidence, tokens).
Errors become plain-language messages (see §9.5).

---

## 8. The agent graph (LangGraph)

Defined in [`agents/rag_agent.py`](agents/rag_agent.py). Each node is a pure
`run(state) -> dict` returning a partial update to `AgentState`.

```
              ┌──────────────── empty & iter<3 ────────────────┐
              ▼                                                 │
   START → query_analyzer → retriever ──┬── sufficient ─────────┼──► context_builder ──► generator → END      (rag_graph)
                               ▲        │                       │          │
                               └────────┴─ insufficient & iter<3┘          └──► END                         (context_graph)
```

| Graph | Ends at | Used by | Why |
|---|---|---|---|
| `rag_graph` | `generator` (blocking `llm.chat`) | `chat.py`, integration test | Simple: one call returns the finished answer |
| `context_graph` | `context_builder` | `webapp.py` `/stream` | Generation happens **outside** the graph so it can stream token by token |

Both are built from the same `_add_retrieval_nodes`, so routing and retries
are identical.

**`AgentState`** ([`state.py`](agents/state.py)):

| Group | Fields |
|---|---|
| Input | `user_query`, `session_id`, `collection_names` (the first one is searched), `source_filenames` |
| Query understanding | `expanded_queries`, `query_intent`, `query_filters`, `requires_table_reasoning`, `requires_image_understanding` |
| Retrieval | `raw_chunks`, `top_chunks`, `context_window` |
| Generation | `messages` (append-only), `final_answer`, `citations`, `confidence_score`, `input_tokens`, `output_tokens` |
| Control | `retrieval_quality`, `iteration_count`, `error` |

`RetrievedChunk` = `chunk_id`, `document_id`, `content`, `content_type`,
`score` (fused RRF), `metadata`.

---

## 9. Web UI and sessions

### 9.1 Routes ([`webapp.py`](webapp.py))

| Route | Method | Does |
|---|---|---|
| `/` | GET | **Discards the previous session** and renders a new one: sidebar (documents, upload, Reset, hints) and chat panel |
| `/upload` | POST | Indexes files (§5) and returns the documents panel |
| `/toggle` | POST | Turns a source on or off for retrieval |
| `/delete` | POST | Removes one document's points and its list entry |
| `/ask` | POST | Checks the caps and returns the user bubble plus the SSE placeholder (§7.2) |
| `/stream` | GET (SSE) | Retrieval, then streamed generation (§7.3) |
| `/reset` | POST | Discards the session and redirects to `/` |

### 9.2 Session model

- Identity is `sid` in a **signed cookie** (Starlette `SessionMiddleware`,
  signed with `SESSION_SECRET`). The cookie also carries `q_count`.
- Per-session state (`docs`, `enabled`, `messages`, `pending`) lives in the
  session store: an in-memory dict, or Redis with `SESSION_BACKEND=redis`
  (JSON per key, `rag:session:<sid>`, **no expiry**: keys stay until the
  session is discarded).
- **Sessions are short-lived by design.** Every load of `/` (a refresh, a new
  tab, or the redirect after Reset) runs `_discard_session`: drop
  `session_<old sid>`, delete the old state, issue a new `sid`. Because every tab
  shares the same cookie, **opening a second tab wipes the first tab's session**.
- The session id isn't shown in the UI.

### 9.3 Citations and the preview dialog

Server side ([`assistant_bubble`](webapp.py)):
1. The answer markdown is rendered to HTML.
2. `_link_citations` finds `[N]` and `[N, M]` markers **outside** `<code>` and
   `<pre>` and replaces each number that matches a real citation with
   `<a class="cite-link" data-ref="N">[N]</a>`. Markers that don't match any
   citation are left as plain text.
3. **Only the sources the answer actually cited** get a chip under "Sources:".
   An answer that cites nothing (a refusal, say) shows no sources.
4. Each chip is a `<button class="cite" data-ref="N">`, followed by a hidden
   `.cite-data` block holding the heading (file · page · sheet) and the excerpt.
   Table excerpts are pre-rendered as HTML tables; prose stays plain text.

Browser side (`_CITE_JS`): one click listener on the document (so it also
works for bubbles HTMX adds later) catches clicks on `.cite` and `.cite-link`,
finds the matching `.cite-data` **in the same bubble**, copies it into the
single `<dialog id="cite-modal">`, and calls `showModal()`. The dialog closes
with ✕, Esc, or a click on the backdrop. While an answer is still streaming,
markers are plain text; they become links when the `done` event arrives.

### 9.4 Sidebar behaviour

- Uploading accepts several files at once (`.pdf .txt .md .csv .xlsx .xls .png .jpg .jpeg .webp`).
- Each document has a checkbox (include or exclude from retrieval) and a delete button.
- Hints say which formats are supported, that a refresh starts a new session,
  and how many questions the demo allows.

### 9.5 Error messages (`_friendly_error`)

| The provider error contains | The user sees |
|---|---|
| `429`, `RESOURCE_EXHAUSTED`, "rate limit" | "The AI service is busy right now (rate limit)…" |
| `402`, "credits" | "The AI service's usage allowance has run out…" |
| `401`, `403` | "The AI service rejected the API key…" |
| anything else | "Something went wrong: <first 300 characters>" |

---

## 10. Grounding and guardrails

- **Context only.** The generator's system prompt forbids answering outside the
  given chunks and requires `[N]` citations. When the context doesn't cover the
  question, the model says so. An end-to-end test confirmed that an off-topic
  question ("capital of Australia") is refused.
- **Scoped retrieval.** Only the session's own collection, and only the checked
  sources, are searched.
- **Safe rendering.** markdown-it's `default` preset escapes raw HTML in model
  output. Citation links are added afterwards, only for real citation numbers.
- **Abuse limits** for the public demo: questions per visitor and per day
  (§7.2). Uploads have no size cap beyond the server's own limits.

---

## 11. Configuration reference

All settings come from [`config/settings.py`](config/settings.py) and can be
overridden in `.env` (case-insensitive). Unknown keys are ignored
(`extra="ignore"`), so leftover keys from older versions (`GENERATION_MODEL`,
`RERANKER_MODEL`, …) do no harm.

| Area | Setting | Default | Meaning |
|---|---|---|---|
| LLM | `LLM_PROVIDER` | `groq` | `groq` or `hf` |
| | `LLM_MODEL` | *(empty)* | Empty means the provider's default: `qwen/qwen3.8-27b` (Groq) or `Qwen/Qwen3.8-27B` (HF) |
| | `GROQ_API_KEY` | — | Required for `groq` |
| | `HF_TOKEN` / `HF_PROVIDER` | — / `auto` | For `hf` (the token is also exported for huggingface_hub) |
| | `MAX_ANSWER_TOKENS` | `1024` | Answer length cap |
| | `MAX_CONTEXT_CHARS` | `16000` | Retrieved-context cap per prompt |
| Embeddings | `EMBEDDING_PROVIDER` | `openrouter` | `openrouter`, `gemini` or `local` |
| | `OPENROUTER_API_KEY` | — | Required for `openrouter` |
| | `OPENROUTER_EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | |
| | `EMBED_CONCURRENCY` | `4` | Parallel embedding requests during ingest |
| | `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | For `gemini` |
| | `EMBEDDING_MODEL` | `all-mpnet-base-v2` | For `local` |
| | `EMBEDDING_DIM` | `768` | Qdrant dense size; vectors are truncated to this |
| Vision | `GOOGLE_API_KEY` | — | Image captions (and Gemini embeddings) |
| | `VISION_MODEL` | `gemini-2.5-flash` | |
| Retrieval | `RETRIEVER_TOP_K` | `20` | Hits per sub-query (each search leg fetches 2×) |
| | `CONTEXT_TOP_N` | `6` | Fused chunks sent to the LLM |
| Qdrant | `QDRANT_LOCAL_PATH` | *(empty; the entry points default it to `./qdrant_local`)* | Folder, `:memory:`, or empty for server mode |
| | `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / — | Server mode |
| Sessions | `SESSION_SECRET` | `dev-insecure-change-me` | **Change this in production** |
| | `SESSION_BACKEND` | `memory` | `memory` or `redis` |
| | `REDIS_URL` | `redis://localhost:6379` | |
| Demo caps | `MAX_QUESTIONS_PER_SESSION` | `20` | Per visitor (0 = off) |
| | `DAILY_QUESTION_LIMIT` | `80` | Per process per day (0 = off) |
| Monitoring | `MLFLOW_ENABLED` | `true` | |
| | `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Relative path, so run from the project root |
| | `MLFLOW_EXPERIMENT_NAME` | `rag-system` | |

---

## 12. Monitoring (MLflow)

[`monitoring/mlflow_tracer.py`](monitoring/mlflow_tracer.py), used by `/stream`:

- `get_tracer(question, sid)` returns a `RAGTracer` when MLflow is enabled and
  configured, otherwise a `NoopTracer` with the same methods that does
  nothing.
- Setup happens **lazily** on the first question, never at import. Every MLflow
  call is wrapped so that failures are swallowed: monitoring can never break a
  question.
- Each question is one MLflow run in experiment `rag-system`. It records:
  - **params**: session id, LLM provider and model, `retriever_top_k`,
    `context_top_n`, embedding model, a preview of the question, intent,
    number of sub-queries, the table/image flags, retrieval quality.
  - **metrics**: `graph_invoke_latency_ms`, `total_latency_ms`,
    `num_raw_chunks`, `retrieval_max_score` and `retrieval_mean_score`,
    `input_tokens`, `output_tokens`, `total_tokens`, `confidence_score`,
    `num_citations`.
  - **artifacts**: `answer.txt`, `citations.json`.
- Only the whole graph's time is measured, not each node. Generation time is
  part of `total_latency_ms`.
- Dashboard (from the project root):
  `.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.

---

## 13. Deployment

### 13.1 Local

```
.venv\Scripts\python.exe webapp.py          # http://localhost:5001  (one process, reload=False)
.venv\Scripts\python.exe chat.py doc.pdf    # CLI
```

Only one process can open `qdrant_local/` at a time. **Restart `webapp.py`
after code changes**, because nothing reloads automatically. A stale process
running old code is what once produced a Hugging Face 402 error even after the
switch to Groq.

### 13.2 Hugging Face Space (Docker), for the public demo

- [`Dockerfile`](Dockerfile): `python:3.11-slim`, plus `tesseract-ocr` from
  apt, a non-root `user` (uid 1000, as Spaces requires), `pip install -r
  requirements.txt` (**no torch**: every model is an API call),
  `QDRANT_LOCAL_PATH=/home/user/app/qdrant_local`, `MLFLOW_ENABLED=false`, and
  `uvicorn webapp:app --host 0.0.0.0 --port 7860 --workers 1`.
- The YAML front-matter at the top of [`README.md`](README.md) (`sdk: docker`,
  `app_port: 7860`) is what makes the repo a Space.
- **Secrets** to set in the Space: `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
  `GOOGLE_API_KEY` (image captions), `SESSION_SECRET`.
- Storage is **temporary**: a restart or sleep wipes `qdrant_local/`. That
  suits short-lived sessions.
- **One worker** is required: embedded Qdrant's file lock, the in-memory session
  store and the per-process daily counter all assume a single process.
- `.dockerignore` keeps `.env`, `.venv`, `qdrant_local`, MLflow data, tests and
  sample documents out of the image.

### 13.3 Scaling out

Run Qdrant as a **server** (leave `QDRANT_LOCAL_PATH` empty and set
`QDRANT_URL`), use `SESSION_BACKEND=redis` with a real `SESSION_SECRET`, then
run `uvicorn webapp:app --workers N`. In this mode the startup cleanup of
`session_*` collections is skipped automatically. The daily question counter
is per process, so the effective daily cap becomes N × `DAILY_QUESTION_LIMIT`.
Move the counter to Redis if that matters.

---

## 14. Failure modes and limits

| Situation | What happens | What to do |
|---|---|---|
| Groq rate limit (429) | Friendly "busy" message for that question | Wait a minute; lower the demo caps; or pay for a higher tier |
| Groq withdraws the preview model | LLM calls fail | Set `LLM_MODEL` to another Groq model, or `LLM_PROVIDER=hf` with credits |
| OpenRouter has no credit (402) or a bad key | Uploads fail for that file; questions fail at the embedding step | Top up OpenRouter or fix the key |
| OpenRouter 429 / 5xx | Retried 4 times with backoff, then fails | Usually clears up after the retries |
| Gemini caption fails | Image is indexed from OCR text alone; fails only if OCR is empty too | Install Tesseract, or check the Gemini quota |
| Tesseract not installed (the dev PC) | Image text depends entirely on Gemini; **scanned PDFs produce no text** ("no readable text") | Install Tesseract and add it to PATH |
| Second browser tab, or a refresh | The previous session's uploads and chat are discarded | Expected behaviour |
| `chat.py` and `webapp.py` at the same time | `AlreadyLocked` | Run one at a time, or use a Qdrant server |
| Port 5001 won't bind after a crash | An orphaned process still holds the socket | Find it with `Get-CimInstance Win32_Process` and `taskkill /F` |
| Very large spreadsheets | One schema chunk plus a chunk per 20 rows; only the top 6 chunks reach the LLM | Questions needing totals over *all* rows can't be answered exactly |
| Sparse search quality | No corpus-wide IDF (common words aren't down-weighted), and hash collisions between words | Acceptable here; a fitted BM25 encoder would improve it |
| Embedded images inside PDFs (diagrams, pictured tables) | Not extracted or indexed, so their content can't be retrieved | Possible addition: caption slide images with the vision model (costs API calls) |

---

## 15. Testing

```
.venv\Scripts\python.exe -m pytest tests/unit -v                         # no network or keys
.venv\Scripts\python.exe -m pytest tests/integration -v -m integration   # live Groq call
```

| Test file | Covers |
|---|---|
| `tests/unit/test_llm.py` | `strip_think`, the streaming `filter_think` (including tags split across chunks), `parse_json` edge cases, query-analyser fallbacks for bad JSON or wrong types |
| `tests/unit/test_retriever_fusion.py` | Scores added across sub-queries, sort order, `top_chunks` cut-off, quality classification, store results not modified |
| `tests/unit/test_openrouter_embedder.py` | Truncation and normalisation, result ordering (including parallel batches finishing out of order), query instruction only on queries, batching, 429 retry, error handling |
| `tests/unit/test_pipeline_batching.py` | A whole PDF is embedded in **one** `embed_batch` call and stored in one write; large documents are written in 256-point slices |
| `tests/unit/test_pdf_table_filter.py` | Layout boxes, mostly-empty grids and duplicate 2-column grids are dropped; real tables are kept |
| `tests/unit/test_chunking_formats.py` | Unique IDs for 20-row windows (the collision bug), no empty schema chunk for PDF tables, prefix-collision safety, CSV loading |
| `tests/unit/test_webapp_citations.py` | `[N]` → links, chips only for cited sources, unknown refs and code left alone, table excerpts as HTML, page load discards the previous session |
| `tests/unit/test_chunker.py` | Text splitting, stable IDs, schema chunk for tables, image chunk |
| `tests/unit/test_sparse_index.py` | Empty input, parallel index/value lists, repeated tokens score higher |
| `tests/integration/test_rag_agent.py` | The full `rag_graph` with mocked Qdrant and embedder and a **live** Groq call: answer is non-empty with no `<think>` text, citations, a valid quality label |

`pytest.ini` sets `testpaths = tests`, `asyncio_mode = auto`, and the
`integration` marker.
