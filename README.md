---
title: RAG Document Q&A
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Upload documents and get cited answers (LangGraph + Qwen)
---

# RAG System

A document question-answering system built on a **LangGraph** agent, with hybrid
retrieval, multi-query fusion, and **Qwen 3.8 27B** (via Groq) for generation. Upload
documents (PDF, text, spreadsheets, images) and ask questions that are answered
**grounded in your content, with citations**.

It runs **in-process — no Docker, no local models**: an embedded Qdrant, Qwen via
Groq for the LLM, Qwen3 Embedding 8B via OpenRouter for embeddings, Gemini for
image captions, and a local SQLite MLflow store for monitoring. Two entry points share the same core: a **web chat UI**
(FastHTML) and a **command-line chat**.

---

## Features

- **Agentic RAG pipeline** (LangGraph): query analysis → retrieval →
  context building → grounded generation, with an automatic retry loop when
  retrieval is weak.
- **Multi-format ingestion**: PDF (text + tables), TXT/MD/CSV, spreadsheets
  (XLSX/XLS), images (OCR via Tesseract + Gemini vision captioning), and SQL
  databases.
- **Hybrid search**: dense semantic vectors (Qwen3 Embedding 8B, truncated to 768-dim) fused
  with a BM25 sparse index via Reciprocal Rank Fusion in **Qdrant**.
- **Multi-query fusion**: the query is expanded into sub-queries; each chunk's
  RRF scores are summed across them, and the top chunks become the context.
- **Grounded answers with citations** — every claim is tied back to a source
  chunk (hover a citation to preview it); the model is instructed never to answer
  beyond the provided context. Answers render as markdown.
- **Per-session isolation** in the web UI: each browser session gets its own
  Qdrant collection, so users never see each other's documents.
- **Observability** via MLflow (per-query tokens, latency, retrieval scores,
  confidence) — local SQLite store, no server required.

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
   question ─► query_analyzer ─► retriever ─► context_builder ─► generator ─► answer + citations
                    (Qwen)          │  ▲      (top-N fused chunks)    (Qwen)
                                    └──┘ retry on weak retrieval
```

The agent graph lives in [`agents/rag_agent.py`](agents/rag_agent.py); each node
is a small, testable function in [`agents/nodes/`](agents/nodes/). For a detailed
step-by-step walkthrough of a query, see the pipeline documentation.

---

## Quick start

### 1. Prerequisites
- **Python 3.11**
- A **Groq API key** (free tier) — https://console.groq.com/keys
- An **OpenRouter API key** with a little prepaid credit (embeddings cost
  about $0.01 per million tokens) — https://openrouter.ai/keys
- Optional: a **Google Gemini API key** for image captions — https://aistudio.google.com/apikey
- Optional: **Tesseract** for OCR of images and scanned PDFs

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
Edit `.env` and set your keys:
```
GROQ_API_KEY=gsk_...          # https://console.groq.com/keys (free tier)
OPENROUTER_API_KEY=sk-or-...  # https://openrouter.ai/keys (needs a little prepaid credit)
GOOGLE_API_KEY=...            # optional: image captions
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
with token counts, latency, retrieval scores, confidence, and the answer
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
| `LLM_PROVIDER`           | `groq`                               | `groq` (free tier) or `hf` (HF Inference Providers) |
| `GROQ_API_KEY`           | —                                    | Groq key (required with `groq`)          |
| `HF_TOKEN` / `HF_PROVIDER` | — / `auto`                         | Only with `LLM_PROVIDER=hf`              |
| `LLM_MODEL`              | _(provider default Qwen 3.8 27B)_    | Override the model id                    |
| `MAX_ANSWER_TOKENS`      | `1024`                               | Answer length cap (Groq free tier)       |
| `MAX_CONTEXT_CHARS`      | `16000`                              | Retrieved-context cap per prompt         |
| `EMBEDDING_PROVIDER`     | `openrouter`                         | `openrouter`, `gemini`, or `local` (needs `requirements-local.txt`) |
| `OPENROUTER_API_KEY`     | —                                    | OpenRouter key (required with `openrouter`) |
| `OPENROUTER_EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b`        | Embedding model (truncated to 768-dim)   |
| `GOOGLE_API_KEY`         | —                                    | Gemini key: image captions               |
| `VISION_MODEL`           | `gemini-2.5-flash`                   | Image captioning during ingest           |
| `RETRIEVER_TOP_K`        | `20`                                 | Hybrid-search hits per sub-query         |
| `CONTEXT_TOP_N`          | `6`                                  | Fused chunks passed to the LLM           |
| `MAX_QUESTIONS_PER_SESSION` | `20`                              | Demo cap per visitor (0 = off)           |
| `DAILY_QUESTION_LIMIT`   | `80`                                 | Demo cap per day, whole app (0 = off)    |
| `QDRANT_LOCAL_PATH`      | _(empty)_                            | Set to run Qdrant in-process (no server) |
| `MLFLOW_TRACKING_URI`    | `sqlite:///mlflow.db`                | MLflow store (local SQLite by default)   |
| `SESSION_BACKEND`        | `memory`                             | Web session store: `memory` or `redis`   |

---

## Deploying to Hugging Face Spaces

The repo is a ready-to-push **Docker Space** (see the front-matter at the top of
this file and the [`Dockerfile`](Dockerfile)). The image has no torch or local
models, so it builds quickly and fits the free CPU tier.

1. Create a Space with SDK **Docker**, then push this repo to it.
2. In the Space's **Settings → Variables and secrets**, add the secrets
   `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY` (image captions), and
   `SESSION_SECRET` (a long random string).
3. The Space builds, then serves the app on port 7860.

Notes: storage is ephemeral (uploads are wiped when the Space restarts or
sleeps), MLflow is off in the image, and `MAX_QUESTIONS_PER_SESSION` /
`DAILY_QUESTION_LIMIT` keep visitors within Groq's free-tier limits.
Each page load starts a fresh session, so visitors' uploads are discarded when
they refresh.

---

## Scaling to production

The web app is Starlette/Uvicorn (ASGI). To run multiple workers: point at a
Qdrant **server** (leave `QDRANT_LOCAL_PATH` empty so `QDRANT_URL` is used —
embedded Qdrant takes an exclusive single-process file lock), set
`SESSION_BACKEND=redis` + a real `SESSION_SECRET`, then run
`uvicorn webapp:app --workers N` instead of the dev `serve()`.

---

## Tech stack

LangGraph · Qwen 3.8 (Groq) · Qwen3 Embedding (OpenRouter) · Google Gemini · Qdrant · FastHTML · MLflow ·
PyMuPDF · Tesseract · Redis (optional, sessions)
