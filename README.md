# Production RAG System

A document question-answering system built on a **LangGraph** agent, with hybrid
retrieval, cross-encoder reranking, and **Google Gemini** for generation. Upload
documents (PDF, text, spreadsheets, images) and ask questions that are answered
**grounded in your content, with citations**.

It ships with three ways to use it: a **web chat UI** (FastHTML), a **command-line
chat**, and a **FastAPI** HTTP service with Swagger docs.

---

## Features

- **Agentic RAG pipeline** (LangGraph): query analysis → retrieval → reranking →
  context building → grounded generation, with an automatic retry loop when
  retrieval is weak.
- **Multi-format ingestion**: PDF (text + tables), TXT/MD/CSV, spreadsheets
  (XLSX/XLS), images (OCR via Tesseract + Gemini vision captioning), and SQL
  databases.
- **Hybrid search**: dense semantic vectors (`all-mpnet-base-v2`) fused with a
  BM25 sparse index via Reciprocal Rank Fusion in **Qdrant**.
- **Cross-encoder reranking** with `BAAI/bge-reranker-v2-m3` for precision.
- **Grounded answers with citations** — every claim is tied back to a source
  chunk; the model is instructed never to answer beyond the provided context.
- **Observability** via MLflow tracing (per-stage latency, tokens, scores).
- **Semantic caching** (Redis) and **async ingestion** (Celery) for the
  production API path.
- **Runs with zero infrastructure** for local use: an embedded, in-process Qdrant
  means no Docker is required to try it.

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
                    (Gemini)        │  ▲          (bge)                        (Gemini)
                                    └──┘ retry on weak retrieval
```

The agent graph lives in [`agents/rag_agent.py`](agents/rag_agent.py); each node
is a small, testable function in [`agents/nodes/`](agents/nodes/).

---

## Quick start

### 1. Prerequisites
- **Python 3.10+** (developed on 3.11)
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
Open **http://localhost:5001**, upload a document on the left, and start asking
questions.

**Command-line chat:**
```bash
python chat.py path/to/document.pdf
```

Both run an **embedded Qdrant** (data persists in `./qdrant_local/`) — no Docker
needed.

---

## Running the full API stack (Docker)

For the production HTTP API with the real Qdrant server, Redis cache, MLflow
tracking, and Celery workers:

```bash
docker compose up --build
```

- API + Swagger docs: **http://localhost:8000/docs**
  - `POST /ingest/upload` — queue a document for ingestion
  - `POST /query` — ask a question
- MLflow UI: **http://localhost:5000**

To view MLflow without Docker, point the tracking URI at a local folder and run
`python -m mlflow ui --backend-store-uri ./mlruns`.

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
cache/          Semantic (Redis) cache
monitoring/     MLflow tracer
tasks/          Celery app and ingestion tasks
api/            FastAPI app and routes
config/         Pydantic settings
tests/          Unit and integration tests
webapp.py       FastHTML web chat UI
chat.py         Command-line chat
```

---

## Configuration reference

All settings live in [`config/settings.py`](config/settings.py) and are
overridable via `.env`:

| Variable             | Default              | Purpose                                   |
|----------------------|----------------------|-------------------------------------------|
| `GOOGLE_API_KEY`     | —                    | Gemini API key (required)                 |
| `GENERATION_MODEL`   | `gemini-2.5-flash`   | Answer generation model                   |
| `ROUTING_MODEL`      | `gemini-2.5-flash`   | Query analysis / vision model             |
| `EMBEDDING_MODEL`    | `all-mpnet-base-v2`  | Dense embedding model (768-dim)           |
| `RERANKER_MODEL`     | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker               |
| `RETRIEVER_TOP_K`    | `20`                 | Candidates retrieved before reranking     |
| `RERANKER_TOP_N`     | `6`                  | Chunks kept after reranking               |
| `QDRANT_LOCAL_PATH`  | _(empty)_            | Set to run Qdrant in-process (no server)  |
| `QDRANT_URL`         | `http://localhost:6333` | Qdrant server URL (when not local)     |

---

## Tech stack

LangGraph · Google Gemini · Qdrant · sentence-transformers · FastAPI · FastHTML ·
Celery · Redis · MLflow · PyMuPDF · Tesseract
