"""
FastHTML chatbot UI for the RAG system.

Upload multiple documents, toggle individual sources in/out of the retrieval
context, delete sources, and reset to a clean per-session state. Each browser
session is isolated: identity rides in a signed cookie and each session gets its
own Qdrant collection, so users never see each other's documents.

Runs entirely in-process by default — embedded (local) Qdrant + the existing
ingestion pipeline + the Gemini-powered LangGraph agent — so no Docker is
required for local use.

Run (local / dev):
    .venv\\Scripts\\python.exe webapp.py
Then open http://localhost:5001

Run (production, horizontally scalable):
    Point at a Qdrant *server* (leave QDRANT_LOCAL_PATH empty so QDRANT_URL is
    used — embedded Qdrant takes an exclusive single-process file lock), set
    SESSION_BACKEND=redis and a real SESSION_SECRET, then:
        uvicorn webapp:app --host 0.0.0.0 --port 5001 --workers 4
"""
import os
from pathlib import Path

# Must be set before any project module imports config.settings.
os.environ.setdefault("QDRANT_LOCAL_PATH", str(Path(__file__).parent / "qdrant_local"))

from uuid import uuid4

from markdown_it import MarkdownIt
from starlette.datastructures import UploadFile
from starlette.responses import Response
from fasthtml.common import *

from config.settings import settings
from ingestion.pipeline import IngestionPipeline
from agents.rag_agent import rag_graph
from vectorstore.qdrant_store import QdrantStore
from web.sessions import get_session_store, collection_for

_pipeline = IngestionPipeline()
_store = QdrantStore()
_sessions = get_session_store()

# `default` preset keeps raw HTML escaped (safe for model output); enable GFM
# tables (compare-intent answers) and strikethrough.
_md = MarkdownIt("default").enable("table").enable("strikethrough")


def _render_md(text: str):
    return NotStr(_md.render(text or ""))

_WELCOME = {
    "role": "assistant",
    "text": "Hi! Upload a document on the left, check it as a source, then ask me anything about it.",
    "citations": [],
    "meta": {},
}

CSS = """
:root{
  --bg:#050505; --panel:#0c0c0d; --panel-2:#161617; --border:#2c2c2e;
  --text:#ffffff; --muted:#9a9a9d; --accent:#ffffff; --accent-2:#d4d4d6;
  --assistant:#141415; --ok:#ededed; --err:#f3f3f3;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;color:var(--text);
  /* shiny black: bright sheen pooled in the corners over near-black */
  background:
    radial-gradient(70% 55% at 0% 0%,    rgba(255,255,255,.22), transparent 62%),
    radial-gradient(70% 55% at 100% 0%,  rgba(255,255,255,.17), transparent 60%),
    radial-gradient(75% 60% at 100% 100%,rgba(255,255,255,.22), transparent 62%),
    radial-gradient(75% 60% at 0% 100%,  rgba(255,255,255,.15), transparent 62%),
    radial-gradient(120% 120% at 50% 50%, transparent 55%, rgba(0,0,0,.55)),
    var(--bg);
  background-attachment:fixed}
.app{display:grid;grid-template-columns:320px 1fr;height:100vh}

/* sidebar */
.sidebar{background:rgba(10,10,11,.72);backdrop-filter:blur(6px);
  border-right:1px solid var(--border);
  padding:22px;display:flex;flex-direction:column;gap:20px;overflow-y:auto}
.brand{font-family:'Press Start 2P',monospace;color:#fff;
  font-size:22px;line-height:1.5;letter-spacing:1px;margin:4px 0 2px;
  text-shadow:0 0 10px rgba(255,255,255,.55),0 0 24px rgba(255,255,255,.25),0 2px 0 #000}
.sidebar h3{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:6px 0 0;font-weight:600}
.sess-id{font-size:11px;color:var(--muted);font-family:ui-monospace,monospace}
.muted{color:var(--muted);font-size:13px}

/* doc rows */
.doc-list{display:flex;flex-direction:column;gap:8px;margin-top:6px}
.doc-row{display:flex;align-items:center;gap:9px;background:var(--panel-2);
  border:1px solid var(--border);padding:8px 11px;border-radius:9px}
.doc-row label{display:flex;align-items:center;gap:9px;flex:1;min-width:0;
  cursor:pointer;font-size:13px}
.doc-row label span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.doc-row input[type=checkbox]{width:15px;height:15px;accent-color:var(--accent);cursor:pointer}
.doc-row.off label span{color:var(--muted);text-decoration:line-through}
.trash{background:none;border:none;color:var(--muted);cursor:pointer;font-size:14px;
  padding:2px 4px;border-radius:6px;line-height:1}
.trash:hover{color:var(--err);background:rgba(245,101,101,.12)}

.upload-form{display:flex;flex-direction:column;gap:11px}
input[type=file]{font-size:12.5px;color:var(--muted);width:100%}
input[type=file]::file-selector-button{background:var(--panel-2);color:var(--text);
  border:1px solid var(--border);padding:7px 11px;border-radius:8px;margin-right:10px;
  cursor:pointer;font-size:12.5px}
.btn{background:linear-gradient(135deg,#ffffff,#cfcfd1);color:#000;
  border:none;padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer;
  font-size:14px;transition:filter .15s,transform .05s}
.btn:hover{filter:brightness(1.06)}
.btn:active{transform:translateY(1px)}
.btn:disabled{opacity:.45;cursor:default}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-ghost:hover{background:rgba(255,255,255,.07);filter:none}
.status{font-size:12.5px;padding:9px 11px;border-radius:9px;margin-top:8px}
.status.ok{background:rgba(255,255,255,.07);color:var(--ok);border:1px solid rgba(255,255,255,.22)}
.status.err{background:rgba(255,255,255,.04);color:var(--err);
  border:1px solid rgba(255,255,255,.45)}
.hint{font-size:12px;color:var(--muted);line-height:1.5;margin-top:auto;
  border-top:1px solid var(--border);padding-top:14px}

/* main */
.main{display:flex;flex-direction:column;height:100vh;min-width:0}
.topbar{padding:17px 28px;border-bottom:1px solid var(--border);font-weight:600;font-size:15px;
  background:rgba(8,8,9,.45)}
.topbar small{display:block;color:var(--muted);font-weight:400;font-size:12px;margin-top:3px}
.messages{flex:1;overflow-y:auto;padding:28px;display:flex;flex-direction:column;gap:18px}
.row{display:flex}
.row.user{justify-content:flex-end}
.row.assistant{justify-content:flex-start}
.bubble{max-width:720px;padding:14px 17px;border-radius:16px;line-height:1.58;
  font-size:15px;word-wrap:break-word}
.bubble.user{background:linear-gradient(135deg,#ffffff,#d4d4d6);
  color:#000;border-bottom-right-radius:5px;white-space:pre-wrap}
.bubble.assistant{background:var(--assistant);border:1px solid var(--border);
  border-bottom-left-radius:5px;white-space:normal}

/* markdown inside assistant bubbles */
.bubble.assistant > *:first-child{margin-top:0}
.bubble.assistant > *:last-child{margin-bottom:0}
.bubble.assistant p{margin:0 0 10px}
.bubble.assistant ul,.bubble.assistant ol{margin:0 0 10px;padding-left:22px}
.bubble.assistant li{margin:3px 0}
.bubble.assistant h1,.bubble.assistant h2,.bubble.assistant h3{margin:14px 0 8px;line-height:1.3}
.bubble.assistant h1{font-size:19px}.bubble.assistant h2{font-size:17px}.bubble.assistant h3{font-size:15px}
.bubble.assistant code{background:rgba(255,255,255,.09);padding:1px 5px;border-radius:5px;
  font-family:ui-monospace,monospace;font-size:13px}
.bubble.assistant pre{background:#000;border:1px solid var(--border);border-radius:9px;
  padding:11px 13px;overflow:auto;margin:0 0 10px}
.bubble.assistant pre code{background:none;padding:0;font-size:12.5px}
.bubble.assistant table{border-collapse:collapse;margin:4px 0 10px;font-size:13px;width:auto}
.bubble.assistant th,.bubble.assistant td{border:1px solid var(--border);padding:6px 10px;text-align:left}
.bubble.assistant th{background:var(--panel-2)}
.bubble.assistant a{color:#fff;text-decoration:underline}
.bubble.assistant blockquote{border-left:3px solid var(--border);margin:0 0 10px;
  padding-left:12px;color:var(--muted)}
.bubble.assistant strong{font-weight:700}

.cites{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}
.cite{position:relative;font-size:11px;color:var(--muted);background:var(--panel);
  border:1px solid var(--border);padding:3px 9px;border-radius:7px;cursor:default}
.cite.has-pop:hover{color:var(--text);border-color:var(--accent)}
.cite-pop{position:absolute;bottom:calc(100% + 8px);left:0;z-index:50;
  width:320px;max-height:240px;overflow:auto;
  background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:10px 12px;box-shadow:0 10px 34px rgba(0,0,0,.65);
  opacity:0;visibility:hidden;transform:translateY(5px);pointer-events:none;
  transition:opacity .12s ease,transform .12s ease}
.cite.has-pop:hover .cite-pop{opacity:1;visibility:visible;transform:translateY(0)}
.cite-pop-head{color:var(--muted);font-size:10.5px;letter-spacing:.03em;
  margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid var(--border)}
.cite-pop-body{color:var(--text);font-size:12px;line-height:1.55;white-space:pre-wrap}
.meta{margin-top:9px;font-size:11px;color:var(--muted);opacity:.85}
.thinking{color:var(--muted);font-size:13px;padding:0 28px 12px;display:none}
.htmx-request.thinking{display:block}
.dots:after{content:'';animation:dots 1.4s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}

/* composer */
.composer{display:flex;gap:11px;padding:18px 28px;border-top:1px solid var(--border);
  background:rgba(8,8,9,.55);backdrop-filter:blur(6px)}
.q-input{flex:1;background:var(--panel-2);border:1px solid var(--border);color:var(--text);
  padding:13px 16px;border-radius:12px;font-size:15px;outline:none;transition:border-color .15s}
.q-input:focus{border-color:var(--accent)}
.q-input::placeholder{color:var(--muted)}

@media(max-width:820px){.app{grid-template-columns:1fr}.sidebar{display:none}}
"""

app, rt = fast_app(
    pico=False,
    secret_key=settings.session_secret,
    hdrs=(
        Link(rel="preconnect", href="https://fonts.googleapis.com"),
        Link(rel="stylesheet",
             href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Press+Start+2P&display=swap"),
        Style(CSS),
    ),
)


# ── session helpers ──────────────────────────────────────────────────────────
def _sid(session) -> str:
    """Ensure the signed cookie carries a session id; return it."""
    if "sid" not in session:
        session["sid"] = uuid4().hex
    return session["sid"]


def _state(session) -> tuple[str, dict, str]:
    sid = _sid(session)
    state = _sessions.get(sid)
    if not state["messages"]:
        state["messages"] = [dict(_WELCOME)]
        _sessions.save(sid, state)
    return sid, state, collection_for(sid)


# ── components ──────────────────────────────────────────────────────────────
def doc_row(name: str, enabled: bool):
    return Div(
        Form(
            Input(type="hidden", name="name", value=name),
            Label(
                Input(type="checkbox", name="checked", value="on", checked=enabled,
                      hx_post="/toggle", hx_target="#docs-inner", hx_swap="outerHTML",
                      hx_trigger="change"),
                Span(name),
            ),
            Button("✕", type="button", cls="trash", title=f"Delete {name}",
                   hx_post="/delete", hx_target="#docs-inner", hx_swap="outerHTML",
                   hx_confirm=f"Delete “{name}”?"),
            cls="doc-row" + ("" if enabled else " off"),
            style="display:flex;align-items:center;gap:9px",
        ),
    )


def docs_inner(state: dict, status=None, ok=True):
    if state["docs"]:
        body = Div(
            *[doc_row(n, state["enabled"].get(n, True)) for n in state["docs"]],
            cls="doc-list",
        )
        hint = Div("Checked sources are included in the answer context.", cls="muted",
                   style="margin-top:8px")
    else:
        body = Div(Span("No documents yet — upload some below.", cls="muted"))
        hint = None
    return Div(
        body,
        hint,
        (Div(status, cls=f"status {'ok' if ok else 'err'}") if status else None),
        id="docs-inner",
    )


def user_bubble(text):
    return Div(Div(text, cls="bubble user"), cls="row user")


def cite_chip(c):
    ref = c.get("ref", "")
    src = c.get("source_file", "?")
    page = c.get("page_num")
    sheet = c.get("sheet_name")
    label = f"{ref} {src}" + (f" · p.{page}" if page else "")
    excerpt = c.get("content")
    if not excerpt:
        return Span(label, cls="cite")
    head = src + (f" · p.{page}" if page else "") + (f" · {sheet}" if sheet else "")
    return Span(
        label,
        Div(
            Div(head, cls="cite-pop-head"),
            Div(excerpt, cls="cite-pop-body"),
            cls="cite-pop",
        ),
        cls="cite has-pop",
    )


def assistant_bubble(text, citations, meta):
    cite_chips = None
    if citations:
        cite_chips = Div(*[cite_chip(c) for c in citations], cls="cites")

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
        Div(_render_md(text), cite_chips, meta_line, cls="bubble assistant"),
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
def index(session):
    sid, state, _ = _state(session)
    return Title("RAG Chat"), Div(
        Div(
            Div("RAG CHAT", cls="brand"),
            Div(f"session_{sid[:8]}", cls="sess-id"),
            Div(
                H3("Documents"),
                docs_inner(state),
                id="docs-panel",
            ),
            Form(
                Input(type="file", name="file", required=True, multiple=True,
                      accept=".pdf,.txt,.md,.csv,.xlsx,.xls,.png,.jpg,.jpeg"),
                Button("Upload & index", cls="btn", type="submit"),
                hx_post="/upload", hx_target="#docs-inner", hx_swap="outerHTML",
                hx_encoding="multipart/form-data", hx_disabled_elt="button",
                cls="upload-form",
            ),
            Button("↺  Reset session", cls="btn btn-ghost",
                   hx_post="/reset", hx_confirm="Delete all documents and start fresh?"),
            Div("Supported: PDF, TXT, MD, CSV, XLSX, images. "
                "Everything runs locally; answers come from Gemini grounded in your files.",
                cls="hint"),
            cls="sidebar",
        ),
        Div(
            Div("Document Q&A",
                Small("Ask questions and get answers cited from your uploaded documents."),
                cls="topbar"),
            Div(*[render_message(m) for m in state["messages"]], id="messages", cls="messages"),
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
async def upload(session, file: list[UploadFile]):
    sid, state, collection = _state(session)
    files = file if isinstance(file, list) else [file]
    indexed, skipped, total = [], [], 0
    try:
        for f in files:
            if not f or not f.filename:
                continue
            if f.filename in state["docs"]:
                skipped.append(f.filename)
                continue
            content = await f.read()
            if not content:
                skipped.append(f.filename)
                continue
            count = _pipeline.ingest_file(
                filename=f.filename, content=content, collection=collection
            )
            state["docs"].append(f.filename)
            state["enabled"][f.filename] = True
            indexed.append(f.filename)
            total += count
        _sessions.save(sid, state)
    except Exception as exc:
        return docs_inner(state, f"Failed to index: {exc}", ok=False)

    if not indexed:
        return docs_inner(state, "Nothing new to index (duplicates or empty).", ok=False)
    msg = f"Indexed {len(indexed)} file(s) — {total} chunk(s)."
    if skipped:
        msg += f" Skipped {len(skipped)} duplicate/empty."
    return docs_inner(state, msg, ok=True)


@rt("/toggle", methods=["POST"])
def toggle(session, name: str, checked: str = ""):
    sid, state, _ = _state(session)
    if name in state["enabled"]:
        state["enabled"][name] = bool(checked)
        _sessions.save(sid, state)
    return docs_inner(state)


@rt("/delete", methods=["POST"])
def delete(session, name: str):
    sid, state, collection = _state(session)
    try:
        _store.delete_by_filename(collection, name)
    except Exception:
        pass
    if name in state["docs"]:
        state["docs"].remove(name)
    state["enabled"].pop(name, None)
    _sessions.save(sid, state)
    return docs_inner(state, f"Deleted “{name}”.", ok=True)


@rt("/ask", methods=["POST"])
def ask(session, question: str):
    sid, state, collection = _state(session)
    selected = [n for n in state["docs"] if state["enabled"].get(n)]

    if not selected:
        warn = "Select at least one source (check a box on the left) before asking."
        state["messages"].append({"role": "user", "text": question})
        state["messages"].append({"role": "assistant", "text": warn, "citations": [], "meta": {}})
        _sessions.save(sid, state)
        return user_bubble(question), assistant_bubble(warn, [], {})

    try:
        result = rag_graph.invoke({
            "user_query":       question,
            "session_id":       sid,
            "collection_names": [collection],
            "source_filenames": selected,
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

    state["messages"].append({"role": "user", "text": question})
    state["messages"].append(
        {"role": "assistant", "text": answer, "citations": citations, "meta": meta}
    )
    _sessions.save(sid, state)
    return user_bubble(question), assistant_bubble(answer, citations, meta)


@rt("/reset", methods=["POST"])
def reset(session):
    sid, _, collection = _state(session)
    try:
        _store.drop_collection(collection)
    except Exception:
        pass
    # New identity → new empty session collection on next load.
    session["sid"] = uuid4().hex
    _sessions.reset(session["sid"])
    return Response(headers={"HX-Redirect": "/"})


if __name__ == "__main__":
    # reload=False is required: the embedded (local) Qdrant takes an exclusive
    # file lock on qdrant_local, and the auto-reloader's second process would
    # collide with it (AlreadyLocked). For multi-worker production, use a Qdrant
    # server + SESSION_BACKEND=redis and run via uvicorn (see module docstring).
    serve(port=5001, reload=False)
