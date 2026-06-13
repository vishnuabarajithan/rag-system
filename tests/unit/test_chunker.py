import pytest
from ingestion.chunkers.semantic_chunker import ChunkingPipeline


@pytest.fixture
def pipeline():
    return ChunkingPipeline(chunk_size=200, chunk_overlap=20)


def test_chunk_text_basic(pipeline):
    text = "Hello world. " * 50
    chunks = pipeline.chunk_text(text, {"filename": "test.txt"})
    assert len(chunks) > 0
    for c in chunks:
        assert c["content_type"] == "text"
        assert "chunk_id" in c
        assert len(c["content"]) <= 250  # chunk_size + small buffer


def test_chunk_text_deduplicates_ids(pipeline):
    text = "Same content block."
    meta = {"filename": "a.txt", "page_num": 1}
    c1 = pipeline.chunk_text(text, meta)
    c2 = pipeline.chunk_text(text, meta)
    assert c1[0]["chunk_id"] == c2[0]["chunk_id"]


def test_chunk_table_includes_schema(pipeline):
    import pandas as pd
    df = pd.DataFrame({"col_a": range(5), "col_b": ["x"] * 5})
    chunks = pipeline.chunk_table("", {"filename": "t.xlsx", "sheet_name": "Sheet1"}, ["col_a", "col_b"], df)
    schema_chunks = [c for c in chunks if c["metadata"].get("is_schema")]
    assert len(schema_chunks) == 1
    assert "col_a" in schema_chunks[0]["content"]


def test_chunk_image(pipeline):
    chunks = pipeline.chunk_image("An image showing a bar chart.", {"filename": "img.png"})
    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "image_caption"
