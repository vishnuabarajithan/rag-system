"""
IngestionPipeline
=================
Orchestrates: load → chunk → embed → upsert to Qdrant.
Supports: PDF, text, spreadsheets, databases, images.
"""
import io
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.text_loader import TextLoader
from ingestion.loaders.spreadsheet_loader import SpreadsheetLoader
from ingestion.loaders.database_loader import DatabaseLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.chunkers.semantic_chunker import ChunkingPipeline
from ingestion.embedders.text_embedder import TextEmbedder
from vectorstore.qdrant_store import QdrantStore
from vectorstore.sparse_index import bm25_encode

_PDF_LOADER    = PDFLoader()
_TEXT_LOADER   = TextLoader()
_SHEET_LOADER  = SpreadsheetLoader()
_IMAGE_LOADER  = ImageLoader()
_CHUNKER       = ChunkingPipeline()
_EMBEDDER      = TextEmbedder()
_STORE         = QdrantStore()

_IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_SHEET_EXTS    = {".xlsx", ".xls", ".ods", ".xlsm"}
_TEXT_EXTS     = {".txt", ".md", ".rst", ".log"}


class IngestionPipeline:

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        collection: str = "default",
    ) -> int:
        """Returns count of upserted chunks."""
        suffix = Path(filename).suffix.lower()
        tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{suffix}"
        tmp.write_bytes(content)

        try:
            if suffix == ".pdf":
                return self._ingest_pdf(tmp, filename, collection)
            elif suffix in _SHEET_EXTS:
                return self._ingest_spreadsheet(tmp, filename, collection)
            elif suffix in _IMAGE_EXTS:
                return self._ingest_image(tmp, filename, collection)
            else:
                return self._ingest_text(tmp, filename, collection)
        finally:
            tmp.unlink(missing_ok=True)

    def ingest_database(
        self,
        connection_string: str,
        table_names: list[str] | None,
        collection: str = "default",
    ) -> int:
        loader = DatabaseLoader(connection_string)
        sheets = loader.load(table_names)
        total = 0
        for sheet in sheets:
            meta = {
                "filename":   connection_string.split("@")[-1],  # hide credentials
                "file_type":  "database",
                "table_name": sheet["table_name"],
                "row_count":  sheet["row_count"],
                "created_at": datetime.utcnow().isoformat(),
            }
            chunks = _CHUNKER.chunk_table(
                markdown="",
                metadata=meta,
                columns=sheet["columns"] if isinstance(sheet["columns"][0], str) else
                        [c["name"] for c in sheet["columns"]],
                df=sheet["dataframe"],
            )
            total += self._embed_and_upsert(chunks, collection)
        return total

    # ── private helpers ────────────────────────────────────────────────────

    def _ingest_pdf(self, path: Path, filename: str, collection: str) -> int:
        pages = _PDF_LOADER.load(path)
        total = 0
        doc_id = str(uuid.uuid4())

        for page in pages:
            base_meta = {
                "document_id": doc_id,
                "filename":    filename,
                "file_type":   "pdf",
                "page_num":    page["page_num"],
                "created_at":  datetime.utcnow().isoformat(),
            }

            if page["text"]:
                chunks = _CHUNKER.chunk_text(page["text"], base_meta)
                total += self._embed_and_upsert(chunks, collection)

            for tbl in page["tables"]:
                table_meta = {**base_meta, "content_type": "table"}
                chunks = _CHUNKER.chunk_table(tbl["markdown"], table_meta, columns=[])
                total += self._embed_and_upsert(chunks, collection)

        return total

    def _ingest_text(self, path: Path, filename: str, collection: str) -> int:
        pages = _TEXT_LOADER.load(path)
        meta = {
            "filename":   filename,
            "file_type":  path.suffix.lstrip("."),
            "page_num":   1,
            "created_at": datetime.utcnow().isoformat(),
        }
        chunks = _CHUNKER.chunk_text(pages[0]["text"], meta)
        return self._embed_and_upsert(chunks, collection)

    def _ingest_spreadsheet(self, path: Path, filename: str, collection: str) -> int:
        sheets = _SHEET_LOADER.load(path)
        total = 0
        for sheet in sheets:
            meta = {
                "filename":   filename,
                "file_type":  path.suffix.lstrip("."),
                "sheet_name": sheet["sheet_name"],
                "row_count":  sheet["row_count"],
                "created_at": datetime.utcnow().isoformat(),
            }
            chunks = _CHUNKER.chunk_table(
                markdown="",
                metadata=meta,
                columns=list(sheet["columns"]),
                df=sheet["dataframe"],
            )
            total += self._embed_and_upsert(chunks, collection)
        return total

    def _ingest_image(self, path: Path, filename: str, collection: str) -> int:
        result = _IMAGE_LOADER.load(path)
        meta = {
            "filename":   filename,
            "file_type":  path.suffix.lstrip("."),
            "created_at": datetime.utcnow().isoformat(),
        }
        chunks = _CHUNKER.chunk_image(result["combined"], meta)
        return self._embed_and_upsert(chunks, collection)

    def _embed_and_upsert(self, chunks: list[dict], collection: str) -> int:
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = _EMBEDDER.embed_batch(texts)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            sparse_indices, sparse_values = bm25_encode(chunk["content"])
            points.append({
                "id":           chunk["chunk_id"],
                "dense":        embedding,
                "sparse":       {"indices": sparse_indices, "values": sparse_values},
                "payload": {
                    "content":      chunk["content"],
                    "content_type": chunk["content_type"],
                    "document_id":  chunk.get("metadata", {}).get("document_id", ""),
                    **chunk.get("metadata", {}),
                },
            })

        _STORE.upsert(collection, points)
        return len(points)
