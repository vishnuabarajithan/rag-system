"""
FastHTML chatbot UI for the RAG system.

Upload multiple documents, toggle individual sources in/out of the retrieval
context, delete sources, and reset to a clean per-session state. Each browser
session is isolated: identity rides in a signed cookie and each session gets its
own Qdrant collection, so users never see each other's documents. Sessions are
ephemeral: every page load (refresh or Reset) discards the previous session's
documents and chat and starts fresh.

Runs entirely in-process by default — embedded (local) Qdrant + the existing
ingestion pipeline + the Qwen-powered LangGraph agent — so no Docker is
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

import re
from datetime import date
from uuid import uuid4

from markdown_it import MarkdownIt
from starlette.datastructures import UploadFile
from starlette.responses import Response
from fasthtml.common import *

from config.settings import settings
from ingestion.pipeline import IngestionPipeline
from agents.rag_agent import context_graph
from agents.nodes.generator import stream_generate
from vectorstore.qdrant_store import QdrantStore
from web.sessions import get_session_store, collection_for
from monitoring.mlflow_tracer import get_tracer

_pipeline = IngestionPipeline()
_store = QdrantStore()
_sessions = get_session_store()

# Embedded Qdrant persists to disk, and sessions are ephemeral: collections
# left by a previous run belong to sessions that no longer exist. Only safe in
# embedded mode (single process) — with a shared Qdrant server, another
# worker's live sessions would be deleted.
if settings.qdrant_local_path:
    _store.drop_collections_with_prefix(collection_for(""))

# Public-demo guard. The per-visitor count lives in the signed cookie, so
# a fresh session (which only rotates `sid`) doesn't clear it. The daily count is
# process-wide — fine for the single-worker deployment.
_daily = {"date": None, "count": 0}


def _limit_reason(session) -> str | None:
    """Return a user-facing message if a question cap is hit, else None."""
    cap = settings.max_questions_per_session
    if cap and session.get("q_count", 0) >= cap:
        return f"You've reached this demo's limit of {cap} questions. Thanks for trying it!"
    today = date.today().isoformat()
    if _daily["date"] != today:
        _daily["date"], _daily["count"] = today, 0
    if settings.daily_question_limit and _daily["count"] >= settings.daily_question_limit:
        return "The demo has reached its daily question limit. Please try again tomorrow."
    return None


def _count_question(session) -> None:
    session["q_count"] = session.get("q_count", 0) + 1
    _daily["count"] += 1


def _friendly_error(exc: Exception) -> str:
    """Turn provider errors into something a visitor can act on."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower():
        return "The AI service is busy right now (rate limit). Please wait a minute and try again."
    if "402" in msg or "credits" in msg.lower():
        return "The AI service's usage allowance has run out. Please try again later."
    if "401" in msg or "403" in msg:
        return "The AI service rejected the API key. Check the server's configuration."
    return f"Something went wrong: {msg[:300]}"

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
/* HTMX puts .htmx-request on the form while /upload runs */
.upload-form .busy{display:none}
.upload-form.htmx-request .idle{display:none}
.upload-form.htmx-request .busy{display:inline}
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

/* citations: inline [N] links + source chips, both open the preview dialog */
.cites{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-top:11px}
.cites-label{font-size:11px;color:var(--muted);margin-right:2px}
.cite{font:inherit;font-size:11px;color:var(--muted);background:var(--panel);
  border:1px solid var(--border);padding:3px 9px;border-radius:7px;cursor:pointer;
  max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cite:hover,.cite:focus-visible{color:var(--text);border-color:var(--accent);outline:none}
.bubble.assistant a.cite-link{color:var(--text);text-decoration:none;font-weight:600;
  font-size:.78em;padding:1px 5px;margin-left:2px;border-radius:6px;cursor:pointer;
  background:rgba(255,255,255,.1);border:1px solid var(--border);white-space:nowrap}
.bubble.assistant a.cite-link:hover{background:rgba(255,255,255,.2);border-color:var(--accent)}
.cite-data{display:none}

.cite-modal{background:var(--panel);color:var(--text);border:1px solid var(--border);
  border-radius:14px;padding:0;width:min(680px,calc(100vw - 32px));max-height:80vh;
  box-shadow:0 20px 60px rgba(0,0,0,.7)}
.cite-modal::backdrop{background:rgba(0,0,0,.6);backdrop-filter:blur(2px)}
.cite-modal-inner{display:flex;flex-direction:column;max-height:80vh}
.cite-modal-top{display:flex;align-items:center;gap:12px;padding:14px 18px;
  border-bottom:1px solid var(--border)}
.cite-modal-ref{font-weight:700;font-size:14px}
.cite-modal-head{flex:1;min-width:0;color:var(--muted);font-size:12.5px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cite-modal-close{background:none;border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:4px 9px;cursor:pointer;font-size:13px}
.cite-modal-close:hover{background:rgba(255,255,255,.08)}
.cite-modal-body{padding:16px 18px;overflow:auto;font-size:13.5px;line-height:1.6}
.cite-modal-body.plain{white-space:pre-wrap}
.cite-modal-body table{border-collapse:collapse;font-size:12.5px}
.cite-modal-body th,.cite-modal-body td{border:1px solid var(--border);padding:5px 9px;text-align:left}
.cite-modal-body th{background:var(--panel-2)}
.cite-modal-body p{margin:0 0 10px}
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
        # HTMX Server-Sent Events extension — drives token-by-token answer streaming.
        Script(src="https://cdn.jsdelivr.net/npm/htmx-ext-sse@2.2.3/dist/sse.js"),
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


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Leave citation-looking text inside code alone.
_CODE_RE = re.compile(r"(<pre[\s\S]*?</pre>|<code[\s\S]*?</code>)")


def _ref_num(c) -> str:
    return (c.get("ref") or "").strip("[]")


def _link_citations(html: str, valid: set[str]) -> tuple[str, set[str]]:
    """Turn [N] / [N, M] markers in rendered answer HTML into clickable links
    (only for refs that exist). Returns the new HTML and the refs used."""
    used: set[str] = set()

    def repl(m):
        nums = [n.strip() for n in m.group(1).split(",")]
        if not all(n in valid for n in nums):
            return m.group(0)
        used.update(nums)
        return "".join(
            f'<a href="#" class="cite-link" data-ref="{n}" title="Preview source {n}">[{n}]</a>'
            for n in nums
        )

    parts = _CODE_RE.split(html)
    for i in range(0, len(parts), 2):  # even indices are outside code
        parts[i] = _CITE_RE.sub(repl, parts[i])
    return "".join(parts), used


def cite_chip(c):
    n = _ref_num(c)
    src = c.get("source_file", "?")
    page = c.get("page_num")
    sheet = c.get("sheet_name")
    where = (f" · p.{page}" if page else "") + (f" · {sheet}" if sheet else "")
    excerpt = c.get("content") or ""
    # Tables are stored as markdown → render them; prose stays as plain text.
    is_table = c.get("content_type") == "table"
    return (
        Button(f"[{n}] {src}{where}", type="button", cls="cite", data_ref=n,
               title="Preview this source"),
        Div(
            Div(src + where, cls="cite-head"),
            Div(_render_md(excerpt) if is_table else excerpt,
                cls="cite-body" + ("" if is_table else " plain")),
            cls="cite-data", data_ref=n,
        ),
    )


def assistant_bubble(text, citations, meta):
    html = _md.render(text or "")
    cite_chips = None
    if citations:
        html, used = _link_citations(html, {_ref_num(c) for c in citations})
        # Only list the sources the answer actually cites.
        cited = [c for c in citations if _ref_num(c) in used]
        if cited:
            cite_chips = Div(Span("Sources:", cls="cites-label"),
                             *[x for c in cited for x in cite_chip(c)], cls="cites")

    meta_line = None
    if meta:
        bits = []
        if meta.get("intent"):
            bits.append(f"intent: {meta['intent']}")
        if meta.get("retrieval"):
            bits.append(f"retrieval: {meta['retrieval']}")
        if meta.get("confidence") is not None:
            bits.append(f"confidence: {meta['confidence']:.2f}")
        if meta.get("tokens"):
            bits.append(f"tokens: {meta['tokens']}")
        if bits:
            meta_line = Div("  ·  ".join(bits), cls="meta")

    return Div(
        Div(NotStr(html), cite_chips, meta_line, cls="bubble assistant"),
        cls="row assistant",
    )


def stream_shell(mid: str):
    """Placeholder assistant row that opens an SSE connection and streams the
    answer into itself. `token` events replace the inner content with the
    markdown rendered so far; the final `done` event swaps the whole row for the
    finished bubble (with citations + meta) and closes the connection."""
    return Div(  # transparent SSE-connection wrapper (display:contents → no layout)
        Div(  # the .row that the final `done` event replaces wholesale
            Div(
                Div(
                    Span(cls="dots"),
                    id=f"stream-{mid}", sse_swap="token", hx_swap="innerHTML",
                ),
                cls="bubble assistant",
            ),
            cls="row assistant",
            sse_swap="done", hx_swap="outerHTML",
        ),
        id=f"asst-{mid}", style="display:contents",
        hx_ext="sse", sse_connect="/stream", sse_close="done",
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
        **{"hx-on::after-request": "if(event.detail.successful) this.reset()"},
        cls="composer",
    )


# ── routes ──────────────────────────────────────────────────────────────────
def _discard_session(session) -> None:
    """Delete the current session's documents + chat, and start a new one."""
    old = session.get("sid")
    if old:
        _store.drop_collection(collection_for(old))
        _sessions.drop(old)
    session["sid"] = uuid4().hex


# Citation preview dialog: chips and inline [N] links (event-delegated, so it
# also works for bubbles swapped in by HTMX) copy the matching hidden
# `.cite-data` block from the same bubble into the dialog.
_CITE_JS = """
document.addEventListener('click', function (e) {
  var dlg = document.getElementById('cite-modal');
  if (e.target === dlg) { dlg.close(); return; }          // backdrop click
  var t = e.target.closest('.cite, .cite-link');
  if (!t) return;
  e.preventDefault();
  var bubble = t.closest('.bubble');
  var data = bubble && bubble.querySelector('.cite-data[data-ref="' + t.dataset.ref + '"]');
  if (!data) return;
  var src = data.querySelector('.cite-body');
  var body = document.getElementById('cite-modal-body');
  document.getElementById('cite-modal-ref').textContent = '[' + t.dataset.ref + ']';
  document.getElementById('cite-modal-head').textContent = data.querySelector('.cite-head').textContent;
  body.innerHTML = src.innerHTML;
  body.className = 'cite-modal-body' + (src.classList.contains('plain') ? ' plain' : '');
  body.scrollTop = 0;
  dlg.showModal();
});
"""


def cite_modal():
    return Dialog(
        Div(
            Div(
                Span(id="cite-modal-ref", cls="cite-modal-ref"),
                Span(id="cite-modal-head", cls="cite-modal-head"),
                Button("✕", type="button", cls="cite-modal-close", title="Close",
                       onclick="this.closest('dialog').close()"),
                cls="cite-modal-top",
            ),
            Div(id="cite-modal-body", cls="cite-modal-body"),
            cls="cite-modal-inner",
        ),
        id="cite-modal", cls="cite-modal",
    )


@rt("/")
def index(session):
    # Every page load (refresh, or the redirect after Reset) starts a fresh
    # session: the previous session's uploaded documents are discarded.
    _discard_session(session)
    sid, state, _ = _state(session)
    return Title("RAG Chat"), Div(
        Div(
            Div("RAG CHAT", cls="brand"),
            Div(
                H3("Documents"),
                docs_inner(state),
                id="docs-panel",
            ),
            Form(
                Input(type="file", name="file", required=True, multiple=True,
                      accept=".pdf,.txt,.md,.csv,.xlsx,.xls,.png,.jpg,.jpeg,.webp"),
                Button(Span("Upload", cls="idle"), Span("Indexing", Span(cls="dots"), cls="busy"),
                       cls="btn", type="submit"),
                hx_post="/upload", hx_target="#docs-inner", hx_swap="outerHTML",
                hx_encoding="multipart/form-data", hx_disabled_elt="button",
                cls="upload-form",
            ),
            Button("↺  Reset session", cls="btn btn-ghost",
                   hx_post="/reset", hx_confirm="Delete all documents and start fresh?"),
            Div("Supported: PDF, TXT, MD, CSV, XLSX, images. "
                "Answers come from Qwen, grounded only in your files. "
                "Refreshing the page starts a new session and discards your uploads.",
                cls="hint"),
            Div(f"Demo: up to {settings.max_questions_per_session} questions per visitor.",
                cls="hint", style="margin-top:0;border-top:none;padding-top:0")
            if settings.max_questions_per_session else None,
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
    ), cite_modal(), Script(
        "function _scrollMsgs(){var m=document.getElementById('messages');"
        "if(m) m.scrollTop=m.scrollHeight;}"
        "document.body.addEventListener('htmx:afterSwap', _scrollMsgs);"
        "document.body.addEventListener('htmx:sseMessage', _scrollMsgs);"
    ), Script(_CITE_JS)


@rt("/upload", methods=["POST"])
async def upload(session, file: list[UploadFile]):
    sid, state, collection = _state(session)
    files = file if isinstance(file, list) else [file]
    indexed, skipped, failed = [], [], []
    # Each file is indexed independently: one bad file doesn't block the rest.
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
        try:
            count = _pipeline.ingest_file(
                filename=f.filename, content=content, collection=collection
            )
        except Exception as exc:
            failed.append(f"{f.filename}: {_friendly_error(exc).removeprefix('Something went wrong: ')}")
            continue
        if not count:
            failed.append(f"{f.filename}: no readable text found")
            continue
        state["docs"].append(f.filename)
        state["enabled"][f.filename] = True
        indexed.append(f.filename)
    _sessions.save(sid, state)

    if failed:
        return docs_inner(state, "Couldn't index " + "; ".join(failed), ok=False)
    if not indexed:
        return docs_inner(state, "Nothing new to index (duplicates or empty).", ok=False)
    # Success is shown by the document appearing in the list — no status banner.
    return docs_inner(state)


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
    """Return immediately: the user's bubble (shown the instant they hit Enter)
    plus a streaming placeholder that connects to /stream. The model runs in
    /stream, not here, so the question appears before the answer starts."""
    sid, state, collection = _state(session)
    selected = [n for n in state["docs"] if state["enabled"].get(n)]

    state["messages"].append({"role": "user", "text": question})

    if not selected:
        warn = "Select at least one source (check a box on the left) before asking."
        state["messages"].append({"role": "assistant", "text": warn, "citations": [], "meta": {}})
        state["pending"] = None
        _sessions.save(sid, state)
        return user_bubble(question), assistant_bubble(warn, [], {})

    limited = _limit_reason(session)
    if limited:
        state["messages"].append({"role": "assistant", "text": limited, "citations": [], "meta": {}})
        state["pending"] = None
        _sessions.save(sid, state)
        return user_bubble(question), assistant_bubble(limited, [], {})
    _count_question(session)

    mid = uuid4().hex[:8]
    state["pending"] = {"question": question, "selected": selected, "mid": mid}
    _sessions.save(sid, state)
    return user_bubble(question), stream_shell(mid)


@rt("/stream")
def stream(session):
    """SSE endpoint (opened by the /ask placeholder). Runs retrieval, then
    streams the answer token-by-token as `token` events, and finishes with a
    `done` event carrying the finished bubble (citations + meta)."""
    sid, state, collection = _state(session)
    pending = state.get("pending")

    def gen():
        if not pending:
            yield sse_message(assistant_bubble("Nothing to answer.", [], {}), event="done")
            return

        question = pending["question"]
        selected = pending["selected"]
        tracer = get_tracer(question, sid)
        parts: list[str] = []
        answer, citations, meta = "", [], {}
        try:
            with tracer.start_run():
                with tracer.stage("graph_invoke"):
                    ctx = context_graph.invoke({
                        "user_query":       question,
                        "session_id":       sid,
                        "collection_names": [collection],
                        "source_filenames": selected,
                        "iteration_count":  0,
                        "messages":         [],
                    })
                citations = ctx.get("citations", [])
                top = ctx.get("top_chunks", [])
                confidence = min(1.0, len(top) / settings.context_top_n) if top else 0.0

                usage: dict = {}
                for delta in stream_generate({**ctx, "user_query": question}, usage):
                    parts.append(delta)
                    # Plain str → sse_message emits the rendered HTML unescaped
                    # (markdown-it already escaped any raw HTML in the source).
                    yield sse_message(_md.render("".join(parts)), event="token")

                answer = "".join(parts) or "I couldn't generate an answer for that."
                in_tok = usage.get("input_tokens", 0) or 0
                out_tok = usage.get("output_tokens", 0) or 0
                meta = {
                    "intent":     ctx.get("query_intent"),
                    "retrieval":  ctx.get("retrieval_quality"),
                    "confidence": confidence,
                    "tokens":     (in_tok + out_tok) or None,
                }
                tracer.log_query_analysis(ctx)
                tracer.log_retrieval(ctx.get("raw_chunks", []), ctx.get("retrieval_quality", ""))
                tracer.log_generation(in_tok, out_tok, confidence)
                tracer.log_final_answer(answer, citations)
        except Exception as exc:
            answer, citations, meta = _friendly_error(exc), [], {}

        state["messages"].append(
            {"role": "assistant", "text": answer, "citations": citations, "meta": meta}
        )
        state["pending"] = None
        _sessions.save(sid, state)
        yield sse_message(assistant_bubble(answer, citations, meta), event="done")

    return EventStream(gen())


@rt("/reset", methods=["POST"])
def reset(session):
    # Discard now (not only on the reload), then reload into a fresh session.
    _discard_session(session)
    return Response(headers={"HX-Redirect": "/"})


if __name__ == "__main__":
    # reload=False is required: the embedded (local) Qdrant takes an exclusive
    # file lock on qdrant_local, and the auto-reloader's second process would
    # collide with it (AlreadyLocked). For multi-worker production, use a Qdrant
    # server + SESSION_BACKEND=redis and run via uvicorn (see module docstring).
    serve(port=5001, reload=False)
