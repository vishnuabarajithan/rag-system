"""
FastHTML chatbot UI for the RAG system.

Upload documents and ask questions about them, with answers grounded in the
uploaded content and shown with citations. Runs entirely in-process — an
embedded (local) Qdrant, the existing ingestion pipeline, and the Gemini-powered
LangGraph agent — so no Docker or external services are required.

Run:
    .venv\\Scripts\\python.exe webapp.py
Then open http://localhost:5001
"""
import os
from pathlib import Path

# Must be set before any project module imports config.settings.
os.environ.setdefault("QDRANT_LOCAL_PATH", str(Path(__file__).parent / "qdrant_local"))

from starlette.datastructures import UploadFile
from fasthtml.common import *

from ingestion.pipeline import IngestionPipeline
from agents.rag_agent import rag_graph

COLLECTION = "default"
_pipeline = IngestionPipeline()

# In-process state for this single-user local tool.
STATE = {
    "docs": [],
    "messages": [
        {
            "role": "assistant",
            "text": "Hi! Upload a document on the left, then ask me anything about it.",
            "citations": [],
            "meta": {},
        }
    ],
}

CSS = """
:root{
  --bg:#0e1016; --panel:#161922; --panel-2:#1e222d; --border:#2a2f3b;
  --text:#e7e9f0; --muted:#9aa2b3; --accent:#7c5cff; --accent-2:#9d7bff;
  --assistant:#20242f; --ok:#7ee2a8; --err:#ff9b9b;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:radial-gradient(1100px 600px at 85% -12%, #1c2030, var(--bg));
  color:var(--text)}
.app{display:grid;grid-template-columns:308px 1fr;height:100vh}

/* sidebar */
.sidebar{background:var(--panel);border-right:1px solid var(--border);
  padding:22px;display:flex;flex-direction:column;gap:20px;overflow-y:auto}
.brand{display:flex;align-items:center;gap:11px;font-weight:700;font-size:18px}
.brand .dot{width:13px;height:13px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 0 18px var(--accent)}
.sidebar h3{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:6px 0 0;font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.chip{background:var(--panel-2);border:1px solid var(--border);padding:6px 11px;
  border-radius:999px;font-size:12px;display:inline-flex;align-items:center;gap:7px}
.chip .fdot{width:7px;height:7px;border-radius:50%;background:var(--accent-2)}
.muted{color:var(--muted);font-size:13px}
.upload-form{display:flex;flex-direction:column;gap:11px}
input[type=file]{font-size:12.5px;color:var(--muted);width:100%}
input[type=file]::file-selector-button{background:var(--panel-2);color:var(--text);
  border:1px solid var(--border);padding:7px 11px;border-radius:8px;margin-right:10px;
  cursor:pointer;font-size:12.5px}
.btn{background:linear-gradient(135deg,var(--accent),var(--accent-2));color:#fff;
  border:none;padding:10px 14px;border-radius:10px;font-weight:600;cursor:pointer;
  font-size:14px;transition:filter .15s}
.btn:hover{filter:brightness(1.09)}
.btn:disabled{opacity:.55;cursor:default}
.status{font-size:12.5px;padding:9px 11px;border-radius:9px;margin-top:4px}
.status.ok{background:rgba(72,187,120,.12);color:var(--ok);border:1px solid rgba(72,187,120,.25)}
.status.err{background:rgba(245,101,101,.12);color:var(--err);border:1px solid rgba(245,101,101,.25)}
.hint{font-size:12px;color:var(--muted);line-height:1.5;margin-top:auto;
  border-top:1px solid var(--border);padding-top:14px}

/* main */
.main{display:flex;flex-direction:column;height:100vh;min-width:0}
.topbar{padding:17px 28px;border-bottom:1px solid var(--border);font-weight:600;font-size:15px}
.topbar small{display:block;color:var(--muted);font-weight:400;font-size:12px;margin-top:3px}
.messages{flex:1;overflow-y:auto;padding:28px;display:flex;flex-direction:column;gap:18px}
.row{display:flex}
.row.user{justify-content:flex-end}
.row.assistant{justify-content:flex-start}
.bubble{max-width:720px;padding:14px 17px;border-radius:16px;line-height:1.58;
  white-space:pre-wrap;font-size:15px;word-wrap:break-word}
.bubble.user{background:linear-gradient(135deg,var(--accent),var(--accent-2));
  color:#fff;border-bottom-right-radius:5px}
.bubble.assistant{background:var(--assistant);border:1px solid var(--border);
  border-bottom-left-radius:5px}
.cites{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}
.cite{font-size:11px;color:var(--muted);background:var(--panel);
  border:1px solid var(--border);padding:3px 9px;border-radius:7px}
.meta{margin-top:9px;font-size:11px;color:var(--muted);opacity:.85}
.thinking{color:var(--muted);font-size:13px;padding:0 28px 12px;display:none}
.htmx-request.thinking{display:block}
.dots:after{content:'';animation:dots 1.4s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}

/* composer */
.composer{display:flex;gap:11px;padding:18px 28px;border-top:1px solid var(--border);
  background:var(--panel)}
.q-input{flex:1;background:var(--panel-2);border:1px solid var(--border);color:var(--text);
  padding:13px 16px;border-radius:12px;font-size:15px;outline:none;transition:border-color .15s}
.q-input:focus{border-color:var(--accent)}
.q-input::placeholder{color:var(--muted)}

@media(max-width:820px){.app{grid-template-columns:1fr}.sidebar{display:none}}
"""

app, rt = fast_app(
    pico=False,
    hdrs=(
        Link(rel="preconnect", href="https://fonts.googleapis.com"),
        Link(rel="stylesheet",
             href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"),
        Style(CSS),
    ),
)


# ── components ──────────────────────────────────────────────────────────────
def doc_chips():
    if not STATE["docs"]:
        return Div(Span("No documents yet", cls="muted"), cls="chips")
    return Div(*[Span(Span(cls="fdot"), d, cls="chip") for d in STATE["docs"]], cls="chips")


def docs_inner(status=None, ok=True):
    return Div(
        doc_chips(),
        (Div(status, cls=f"status {'ok' if ok else 'err'}") if status else None),
        id="docs-inner",
    )


def user_bubble(text):
    return Div(Div(text, cls="bubble user"), cls="row user")


def assistant_bubble(text, citations, meta):
    cite_chips = None
    if citations:
        items = []
        for c in citations:
            ref = c.get("ref", "")
            src = c.get("source_file", "?")
            page = c.get("page_num")
            label = f"{ref} {src}" + (f" · p.{page}" if page else "")
            items.append(Span(label, cls="cite"))
        cite_chips = Div(*items, cls="cites")

    meta_line = None
    if meta:
        bits = []
        if meta.get("intent"):
            bits.append(f"intent: {meta['intent']}")
        if meta.get("retrieval"):
            bits.append(f"retrieval: {meta['retrieval']}")
        if meta.get("confidence") is not None:
            bits.append(f"confidence: {meta['confidence']:.2f}")
        if bits:
            meta_line = Div("  ·  ".join(bits), cls="meta")

    return Div(
        Div(text, cite_chips, meta_line, cls="bubble assistant"),
        cls="row assistant",
    )


def render_message(m):
    if m["role"] == "user":
        return user_bubble(m["text"])
    return assistant_bubble(m["text"], m.get("citations", []), m.get("meta", {}))


def composer():
    return Form(
        Input(name="question", placeholder="Ask a question about your documents…",
              autocomplete="off", required=True, cls="q-input"),
        Button("Send", cls="btn", type="submit"),
        hx_post="/ask", hx_target="#messages", hx_swap="beforeend",
        hx_indicator="#thinking", hx_disabled_elt="button",
        **{"hx-on::after-request": "if(event.detail.successful) this.reset()"},
        cls="composer",
    )


# ── routes ──────────────────────────────────────────────────────────────────
@rt("/")
def index():
    return Title("RAG Chat"), Div(
        Div(
            Div(Span(cls="dot"), "RAG Chat", cls="brand"),
            Div(
                H3("Documents"),
                docs_inner(),
                id="docs-panel",
            ),
            Form(
                Input(type="file", name="file", required=True,
                      accept=".pdf,.txt,.md,.csv,.xlsx,.xls,.png,.jpg,.jpeg"),
                Button("Upload & index", cls="btn", type="submit"),
                hx_post="/upload", hx_target="#docs-inner", hx_swap="outerHTML",
                hx_encoding="multipart/form-data", hx_disabled_elt="button",
                cls="upload-form",
            ),
            Div("Supported: PDF, TXT, MD, CSV, XLSX, images. "
                "Everything runs locally; answers come from Gemini grounded in your files.",
                cls="hint"),
            cls="sidebar",
        ),
        Div(
            Div("Document Q&A",
                Small("Ask questions and get answers cited from your uploaded documents."),
                cls="topbar"),
            Div(*[render_message(m) for m in STATE["messages"]], id="messages", cls="messages"),
            Div("Thinking", Span(cls="dots"), cls="thinking", id="thinking"),
            composer(),
            cls="main",
        ),
        cls="app",
    ), Script(
        "document.body.addEventListener('htmx:afterSwap', function(){"
        "var m=document.getElementById('messages'); if(m) m.scrollTop=m.scrollHeight;});"
    )


@rt("/upload", methods=["POST"])
async def upload(file: UploadFile):
    try:
        content = await file.read()
        if not content:
            return docs_inner("That file was empty.", ok=False)
        count = _pipeline.ingest_file(
            filename=file.filename, content=content, collection=COLLECTION
        )
        if file.filename not in STATE["docs"]:
            STATE["docs"].append(file.filename)
        return docs_inner(f"Indexed “{file.filename}” — {count} chunk(s).", ok=True)
    except Exception as exc:
        return docs_inner(f"Failed to index: {exc}", ok=False)


@rt("/ask", methods=["POST"])
def ask(question: str):
    try:
        result = rag_graph.invoke({
            "user_query":       question,
            "session_id":       "web",
            "collection_names": [COLLECTION],
            "iteration_count":  0,
            "messages":         [],
        })
        answer = result.get("final_answer") or "I couldn't generate an answer for that."
        citations = result.get("citations", [])
        meta = {
            "intent":     result.get("query_intent"),
            "retrieval":  result.get("retrieval_quality"),
            "confidence": result.get("confidence_score"),
        }
    except Exception as exc:
        answer, citations, meta = f"Something went wrong: {exc}", [], {}

    STATE["messages"].append({"role": "user", "text": question})
    STATE["messages"].append(
        {"role": "assistant", "text": answer, "citations": citations, "meta": meta}
    )
    return user_bubble(question), assistant_bubble(answer, citations, meta)


if __name__ == "__main__":
    # reload=False is required: the embedded (local) Qdrant takes an exclusive
    # file lock on qdrant_local, and the auto-reloader's second process would
    # collide with it (AlreadyLocked).
    serve(port=5001, reload=False)
