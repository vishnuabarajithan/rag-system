from agents.state import AgentState
from config.settings import settings

# Excerpt length kept per citation for the UI preview popup.
_EXCERPT_CHARS = 1500


def run(state: AgentState) -> dict:
    chunks = state.get("top_chunks", [])
    citations: list[dict] = []
    parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        ref = f"[{i + 1}]"
        ctype = chunk.get("content_type", "text")
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "unknown")

        if ctype == "table":
            block = (
                f"{ref} [TABLE from {filename}"
                f", sheet={meta.get('sheet_name', '?')}"
                f", rows={meta.get('row_range', '?')}]\n"
                f"{chunk['content']}"
            )
        elif ctype == "image_caption":
            block = (
                f"{ref} [IMAGE from {filename}]\n"
                f"{chunk['content']}"
            )
        else:
            block = (
                f"{ref} [Source: {filename}"
                f", p.{meta.get('page_num', '?')}]\n"
                f"{chunk['content']}"
            )

        if total_chars + len(block) > settings.max_context_chars:
            break

        parts.append(block)
        total_chars += len(block)
        raw = chunk.get("content") or ""
        excerpt = raw[:_EXCERPT_CHARS] + ("…" if len(raw) > _EXCERPT_CHARS else "")
        citations.append({
            "ref": ref,
            "chunk_id": chunk.get("chunk_id"),
            "source_file": filename,
            "page_num": meta.get("page_num"),
            "sheet_name": meta.get("sheet_name"),
            "content_type": ctype,
            "content": excerpt,
        })

    context = "\n\n---\n\n".join(parts)
    return {"context_window": context, "citations": citations}
