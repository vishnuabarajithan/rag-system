import os

os.environ.setdefault("QDRANT_LOCAL_PATH", ":memory:")

from pathlib import Path
from unittest.mock import MagicMock

from ingestion import pipeline as pl


def test_pdf_is_embedded_and_stored_in_one_pass(monkeypatch):
    pages = [
        {"page_num": n, "text": f"Page {n} text. " * 60, "images": [],
         "tables": [{"markdown": "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", "bbox": []}]}
        for n in (1, 2, 3)
    ]
    loader = MagicMock(); loader.load.return_value = pages
    embedder = MagicMock(); embedder.embed_batch.side_effect = lambda t: [[0.0] * 768 for _ in t]
    store = MagicMock()
    monkeypatch.setattr(pl, "_PDF_LOADER", loader)
    monkeypatch.setattr(pl, "_EMBEDDER", embedder)
    monkeypatch.setattr(pl, "_STORE", store)

    count = pl.IngestionPipeline()._ingest_pdf(Path("x.pdf"), "x.pdf", "c")

    assert embedder.embed_batch.call_count == 1           # was one call per page + table
    sent = embedder.embed_batch.call_args.args[0]
    assert len(sent) == count > 6                          # every page's chunks + 3 tables
    assert store.upsert.call_count == 1


def test_large_document_upserts_in_slices(monkeypatch):
    chunks = [{"chunk_id": str(i), "content": f"c{i}", "content_type": "text", "metadata": {}}
              for i in range(pl._UPSERT_BATCH + 10)]
    embedder = MagicMock(); embedder.embed_batch.side_effect = lambda t: [[0.0] * 768 for _ in t]
    store = MagicMock()
    monkeypatch.setattr(pl, "_EMBEDDER", embedder)
    monkeypatch.setattr(pl, "_STORE", store)

    assert pl.IngestionPipeline()._embed_and_upsert(chunks, "c") == len(chunks)
    assert embedder.embed_batch.call_count == 1
    assert [len(c.args[1]) for c in store.upsert.call_args_list] == [pl._UPSERT_BATCH, 10]
