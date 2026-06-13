"""
Streamlit PoC for the RAG system.

Upload multiple documents, toggle individual sources in/out of the retrieval
context with checkboxes, delete sources, and reset to a clean per-session state.
Runs entirely in-process — embedded (local) Qdrant + the existing ingestion
pipeline + the Gemini-powered LangGraph agent — so no Docker is required.

Run:
    .venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""
import os
import uuid
from pathlib import Path

# Must be set before any project module imports config.settings.
os.environ.setdefault("QDRANT_LOCAL_PATH", str(Path(__file__).parent / "qdrant_local"))

import streamlit as st

from config.settings import settings


@st.cache_resource
def _backend():
    """Load the heavy singletons (embedder, reranker, graph) once per process."""
    from ingestion.pipeline import IngestionPipeline
    from agents.rag_agent import rag_graph
    from vectorstore.qdrant_store import QdrantStore
    return IngestionPipeline(), rag_graph, QdrantStore()


pipeline, rag_graph, store = _backend()

st.set_page_config(page_title="RAG PoC", page_icon="📚", layout="wide")


# ── per-session state ──────────────────────────────────────────────────────
def _init_session() -> None:
    st.session_state["collection"] = f"session_{uuid.uuid4().hex[:8]}"
    st.session_state["docs"] = []          # ordered list of ingested filenames
    st.session_state["enabled"] = {}       # filename -> bool (include in context)
    st.session_state["messages"] = []      # [{role, text, citations, meta}]


if "collection" not in st.session_state:
    _init_session()

collection = st.session_state["collection"]


# ── sidebar: sources ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 Sources")
    st.caption(f"Session collection: `{collection}`")

    if not settings.google_api_key:
        key = st.text_input(
            "GOOGLE_API_KEY (required to answer)",
            type="password",
            help="Not found in .env. Paste a key to enable generation.",
        )
        if key:
            settings.google_api_key = key  # Gemini nodes read this lazily at first call

    uploads = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "csv", "xlsx", "xls", "png", "jpg", "jpeg"],
    )
    if uploads and st.button("Ingest", type="primary", use_container_width=True):
        for f in uploads:
            if f.name in st.session_state["docs"]:
                st.warning(f"`{f.name}` already ingested — skipped.")
                continue
            with st.spinner(f"Ingesting {f.name}…"):
                count = pipeline.ingest_file(
                    filename=f.name, content=f.getvalue(), collection=collection
                )
            st.session_state["docs"].append(f.name)
            st.session_state["enabled"][f.name] = True
            st.success(f"`{f.name}` → {count} chunk(s)")

    st.divider()

    if st.session_state["docs"]:
        st.caption("Check sources to include in the answer context.")
        for name in list(st.session_state["docs"]):
            row, btn = st.columns([0.8, 0.2])
            with row:
                st.session_state["enabled"][name] = st.checkbox(
                    name, value=st.session_state["enabled"].get(name, True), key=f"chk_{name}"
                )
            with btn:
                if st.button("🗑", key=f"del_{name}", help=f"Delete {name}"):
                    store.delete_by_filename(collection, name)
                    st.session_state["docs"].remove(name)
                    st.session_state["enabled"].pop(name, None)
                    st.rerun()
    else:
        st.info("No documents yet. Upload some above.")

    st.divider()
    if st.button("♻️ Reset session", use_container_width=True):
        store.drop_collection(collection)
        _init_session()
        st.rerun()


# ── main: chat ─────────────────────────────────────────────────────────────
st.title("RAG PoC")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        for c in msg.get("citations", []):
            page = f" · p.{c['page_num']}" if c.get("page_num") else ""
            sheet = f" · {c['sheet_name']}" if c.get("sheet_name") else ""
            st.caption(f"{c.get('ref', '-')} {c.get('source_file', '?')}{page}{sheet}")
        meta = msg.get("meta") or {}
        if meta:
            st.caption(
                f"intent={meta.get('intent')} · retrieval={meta.get('retrieval')} · "
                f"confidence={(meta.get('confidence') or 0):.2f}"
            )

if prompt := st.chat_input("Ask a question about your sources…"):
    selected = [n for n in st.session_state["docs"] if st.session_state["enabled"].get(n)]

    st.session_state["messages"].append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not selected:
        warn = "Select at least one source (check a box on the left) before asking."
        st.session_state["messages"].append({"role": "assistant", "text": warn})
        with st.chat_message("assistant"):
            st.info(warn)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = rag_graph.invoke({
                        "user_query":       prompt,
                        "session_id":       collection,
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

            st.markdown(answer)
            for c in citations:
                page = f" · p.{c['page_num']}" if c.get("page_num") else ""
                sheet = f" · {c['sheet_name']}" if c.get("sheet_name") else ""
                st.caption(f"{c.get('ref', '-')} {c.get('source_file', '?')}{page}{sheet}")
            if meta:
                st.caption(
                    f"intent={meta.get('intent')} · retrieval={meta.get('retrieval')} · "
                    f"confidence={(meta.get('confidence') or 0):.2f}"
                )

        st.session_state["messages"].append(
            {"role": "assistant", "text": answer, "citations": citations, "meta": meta}
        )
