from agents.state import AgentState

_MAX_CONTEXT_CHARS = 80_000  # ≈ 20k tokens


def run(state: AgentState) -> dict:
    chunks = state.get("reranked_chunks", [])
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

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total_chars += len(block)
        citations.append({
            "ref": ref,
            "chunk_id": chunk.get("chunk_id"),
            "source_file": filename,
            "page_num": meta.get("page_num"),
            "sheet_name": meta.get("sheet_name"),
            "content_type": ctype,
        })

    context = "\n\n---\n\n".join(parts)
    return {"context_window": context, "citations": citations}
