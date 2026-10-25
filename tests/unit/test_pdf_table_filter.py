import pandas as pd

from ingestion.loaders.pdf_loader import is_real_table


def test_drops_tiny_layout_boxes():
    assert not is_real_table(pd.DataFrame({"BERT and encoder-only models": ["x"]}), "")
    assert not is_real_table(pd.DataFrame({"a": ["1"], "b": ["2"]}), "")      # 1 row


def test_drops_mostly_empty_grids():
    df = pd.DataFrame({"Col0": ["", "Chinchilla law."], "Col1": ["", ""], "x 20": ["", ""]})
    assert not is_real_table(df, "Chinchilla law. x 20")


def test_drops_two_column_grid_that_duplicates_page_text():
    df = pd.DataFrame({"ICL = In-Context Learning": ["Zero-shot", "Few-shot"],
                       "Col1": ["Q: How old", "A: 4."]})
    page = "ICL = In-Context Learning Zero-shot Few-shot Q: How old A: 4."
    assert not is_real_table(df, page)


def test_keeps_real_numeric_table_even_if_text_is_on_page():
    df = pd.DataFrame({"Parameters": ["400M", "1B", "10B"],
                       "FLOPs": ["1.92e19", "1.21e20", "1.23e22"],
                       "Tokens": ["8.0B", "20.2B", "205.1B"]})
    page = "Parameters FLOPs Tokens 400M 1.92e19 8.0B 1B 1.21e20 20.2B 10B 1.23e22 205.1B"
    assert is_real_table(df, page)


def test_keeps_two_column_table_with_new_content():
    df = pd.DataFrame({"Region": ["North", "South", "East"], "Sales": ["1.5M", "2.7M", "0.9M"]})
    assert is_real_table(df, "Quarterly sales by region are shown below.")
