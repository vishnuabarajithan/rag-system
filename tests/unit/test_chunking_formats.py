from pathlib import Path

import pandas as pd

from ingestion.chunkers.semantic_chunker import ChunkingPipeline
from ingestion.loaders.spreadsheet_loader import SpreadsheetLoader


def test_table_windows_get_unique_ids():
    # 45 rows → 3 windows sharing the same header; they used to collide.
    df = pd.DataFrame({"region": ["north"] * 45, "sales": range(45)})
    meta = {"filename": "s.xlsx", "sheet_name": "Sales", "row_count": 45}
    chunks = ChunkingPipeline().chunk_table("", meta, columns=list(df.columns), df=df)
    ids = [c["chunk_id"] for c in chunks]
    assert len(chunks) == 4                      # schema + 3 windows
    assert len(set(ids)) == len(ids)
    assert chunks[1]["content"].startswith("Table: Sales (from s.xlsx), rows 1-20")
    assert chunks[3]["metadata"]["row_range"] == "41-45"


def test_pdf_table_without_columns_has_no_empty_schema_chunk():
    chunks = ChunkingPipeline().chunk_table("| a |\n|---|\n| 1 |", {"filename": "d.pdf"}, columns=[])
    assert len(chunks) == 1 and chunks[0]["content"].startswith("| a |")


def test_text_chunks_with_same_prefix_get_unique_ids():
    pipe = ChunkingPipeline()
    meta = {"filename": "t.txt", "page_num": 1}
    a = pipe._make_id("Disclaimer: " * 10 + "first", meta)
    b = pipe._make_id("Disclaimer: " * 10 + "second", meta)
    assert a != b


def test_csv_loads_as_table(tmp_path: Path):
    p = tmp_path / "people.csv"
    p.write_text("name,age\nada,36\nalan,41\n", encoding="utf-8")
    sheets = SpreadsheetLoader().load(p)
    assert sheets[0]["sheet_name"] == "people"
    assert sheets[0]["columns"] == ["name", "age"] and sheets[0]["row_count"] == 2
