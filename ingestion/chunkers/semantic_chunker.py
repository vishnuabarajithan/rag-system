import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingPipeline:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        splits = self._splitter.split_text(text)
        return [
            {
                "chunk_id":    self._make_id(s, metadata),
                "content":     s,
                "content_type":"text",
                "metadata":    metadata,
            }
            for s in splits if s.strip()
        ]

    def chunk_table(
        self,
        markdown: str,
        metadata: dict,
        columns: list[str],
        df=None,
    ) -> list[dict]:
        chunks = []
        table_name = metadata.get("sheet_name") or metadata.get("table_name", "?")

        schema_text = (
            f"Table: {table_name}\n"
            f"Columns: {', '.join(str(c) for c in columns)}\n"
            f"Row count: {metadata.get('row_count', '?')}"
        )
        chunks.append({
            "chunk_id":    self._make_id(schema_text, metadata),
            "content":     schema_text,
            "content_type":"table",
            "metadata":    {**metadata, "is_schema": True},
        })

        if df is not None and len(df) > 0:
            WINDOW = 20
            for start in range(0, len(df), WINDOW):
                window = df.iloc[start:start + WINDOW]
                window_md = window.to_markdown(index=False)
                row_meta = {**metadata, "row_range": f"{start}-{start + len(window)}"}
                chunks.append({
                    "chunk_id":    self._make_id(window_md, row_meta),
                    "content":     window_md,
                    "content_type":"table",
                    "metadata":    row_meta,
                })
        elif markdown:
            chunks.append({
                "chunk_id":    self._make_id(markdown, metadata),
                "content":     markdown,
                "content_type":"table",
                "metadata":    metadata,
            })

        return chunks

    def chunk_image(self, combined_text: str, metadata: dict) -> list[dict]:
        return [{
            "chunk_id":    self._make_id(combined_text, metadata),
            "content":     combined_text,
            "content_type":"image_caption",
            "metadata":    metadata,
        }]

    @staticmethod
    def _make_id(text: str, meta: dict) -> str:
        # Qdrant point IDs must be a UUID or unsigned int. uuid5 gives a
        # deterministic UUID so identical content keeps a stable id.
        key = f"{meta.get('filename','')}-{meta.get('page_num','')}-{text[:64]}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
