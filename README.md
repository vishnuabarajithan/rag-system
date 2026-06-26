# RAG System

A document question-answering system built on a **LangGraph** agent, with hybrid
retrieval, cross-encoder reranking, and **Google Gemini** for generation. Upload
documents (PDF, text, spreadsheets, images) and ask questions that are answered
**grounded in your content, with citations**.

It runs entirely **in-process — no Docker, no external services**: an embedded
Qdrant, the Gemini API for embeddings + generation, and a local SQLite MLflow
store for monitoring. Two entry points share the same core: a **web chat UI**
(FastHTML) and a **command-line chat**.

---

## Features

- **Agentic RAG pipeline** (LangGraph): query analysis → retrieval → reranking →
  context building → grounded generation, with an automatic retry loop when
  retrieval is weak.
- **Multi-format ingestion**: PDF (text + tables), TXT/MD/CSV, spreadsheets
  (XLSX/XLS), images (OCR via Tesseract + Gemini vision captioning), and SQL
  databases.
- **Hybrid search**: dense semantic vectors (Gemini embeddings, 768-dim) fused
  with a BM25 sparse index via Reciprocal Rank Fusion in **Qdrant**.
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for precision.
- **Grounded answers with citations** — every claim is tied back to a source
  chunk (hover a citation to preview it); the model is instructed never to answer
  beyond the provided context. Answers render as markdown.
- **Per-session isolation** in the web UI: each browser session gets its own
  Qdrant collection, so users never see each other's documents.
- **Observability** via MLflow (per-query tokens, latency, retrieval/rerank
  scores, confidence) — local SQLite store, no server required.

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
   document ──► │  Ingestion: load → chunk → embed → upsert     │
                └──────────────────────────────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────┐
                          │   Qdrant (hybrid)   │  dense + BM25 sparse
                          └────────────────────┘
                                     ▲
                                     │
   question ─► query_analyzer ─► retriever ─► reranker ─► context_builder ─► generator ─► answer + citations
                    (Gemini)        │  ▲      (MiniLM)                         (Gemini)
                                    └──┘ retry on weak retrieval
```

The agent graph lives in [`agents/rag_agent.py`](agents/rag_agent.py); each node
is a small, testable function in [`agents/nodes/`](agents/nodes/). For a detailed
step-by-step walkthrough of a query, see the pipeline documentation.

---

## Quick start

### 1. Prerequisites
- **Python 3.11**
- A **Google Gemini API key** — get one at https://aistudio.google.com/apikey

### 2. Set up the environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env      # Windows: copy .env.example .env
```
Edit `.env` and set your key:
```
GOOGLE_API_KEY=your-key-here
```

### 4. Run it

**Web UI (recommended):**
```bash
python webapp.py
```
Open **http://localhost:5001**, upload a document on the left, check it as a
source, and start asking questions.

**Command-line chat:**
```bash
python chat.py path/to/document.pdf   # ingest then loop Q&A
python chat.py                        # query what's already stored
```

Both run an **embedded Qdrant** (data persists in `./qdrant_local/`) — no Docker
needed.

---

## Monitoring (MLflow)

Every query is traced to a local SQLite store (`mlflow.db`). To view the
dashboard, run from the project root in a second terminal:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open **http://localhost:5000** → experiment `rag-system`: one run per question
with token counts, latency, retrieval/rerank scores, confidence, and the answer
saved as an artifact. Set `MLFLOW_ENABLED=false` to turn tracing off.

---

## Testing

```bash
pytest tests/unit -v                          # fast, no API key needed
pytest tests/integration -v -m integration    # runs the live agent (needs key)
```

---

## Project structure

```
agents/         LangGraph agent + pipeline nodes (query analysis, retrieval, …)
ingestion/      Loaders, semantic chunker, embedders, ingestion pipeline
vectorstore/    Qdrant store, collection setup, BM25 sparse index
reranking/      Cross-encoder reranker
monitoring/     MLflow tracer
web/            Per-session state store (in-memory / Redis)
config/         Pydantic settings
tests/          Unit and integration tests
webapp.py       FastHTML web chat UI
chat.py         Command-line chat
```

---

## Configuration reference

All settings live in [`config/settings.py`](config/settings.py) and are
overridable via `.env`:

| Variable                 | Default                              | Purpose                                  |
|--------------------------|--------------------------------------|------------------------------------------|
| `GOOGLE_API_KEY`         | —                                    | Gemini API key (required)                |
| `GENERATION_MODEL`       | `gemini-2.5-flash`                   | Answer generation model                  |
| `ROUTING_MODEL`          | `gemini-2.5-flash`                   | Query analysis / vision model            |
| `EMBEDDING_PROVIDER`     | `gemini`                             | `gemini` (API) or `local` (mpnet on CPU) |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001`               | Dense embedding model (768-dim)          |
| `RERANKER_MODEL`         | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker                 |
| `RETRIEVER_TOP_K`        | `20`                                 | Candidates retrieved before reranking    |
| `RERANKER_TOP_N`         | `6`                                  | Chunks kept after reranking              |
| `QDRANT_LOCAL_PATH`      | _(empty)_                            | Set to run Qdrant in-process (no server) |
| `MLFLOW_TRACKING_URI`    | `sqlite:///mlflow.db`                | MLflow store (local SQLite by default)   |
| `SESSION_BACKEND`        | `memory`                             | Web session store: `memory` or `redis`   |

---

## Scaling to production

The web app is Starlette/Uvicorn (ASGI). To run multiple workers: point at a
Qdrant **server** (leave `QDRANT_LOCAL_PATH` empty so `QDRANT_URL` is used —
embedded Qdrant takes an exclusive single-process file lock), set
`SESSION_BACKEND=redis` + a real `SESSION_SECRET`, then run
`uvicorn webapp:app --workers N` instead of the dev `serve()`.

---

## Tech stack

LangGraph · Google Gemini · Qdrant · sentence-transformers · FastHTML · MLflow ·
PyMuPDF · Tesseract · Redis (optional, sessions)
