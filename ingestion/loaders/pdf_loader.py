import io
import re
from pathlib import Path
import fitz  # pymupdf


def _ocr_page(page) -> str:
    """OCR a page with no text layer (a scanned PDF). Empty string if
    Tesseract isn't installed or OCR fails."""
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png")))).strip()
    except Exception:
        return ""


_EMPTY_CELL = re.compile(r"^(|none|nan|col\d+)$", re.IGNORECASE)
_WORD = re.compile(r"\w+")


def is_real_table(df, page_text: str) -> bool:
    """Reject the "tables" find_tables() reports on slides and layout-heavy
    pages: layout boxes with few cells, mostly-empty grids, or grids whose
    words all already appear in the page text (they'd only duplicate it)."""
    rows, cols = df.shape
    if rows < 2 or cols < 2:
        return False

    cells = [str(c).strip() for c in df.columns] + [str(v).strip() for v in df.values.ravel()]
    filled = [c for c in cells if not _EMPTY_CELL.match(c)]
    if len(filled) < 0.5 * len(cells):
        return False

    table_words = _WORD.findall(" ".join(filled).lower())
    if not table_words:
        return False
    page_words = set(_WORD.findall(page_text.lower()))
    covered = sum(1 for w in table_words if w in page_words) / len(table_words)
    # Numeric tables also appear in the page text, so only drop a duplicate
    # when it adds no structure worth keeping (≤ 2 columns).
    return not (covered >= 0.9 and cols <= 2)


class PDFLoader:
    """
    Extracts text and tables (as markdown) from a PDF.
    Returns one dict per page.
    """

    def load(self, path: Path) -> list[dict]:
        doc = fitz.open(str(path))
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                text = _ocr_page(page)

            tables = []
            for tbl in page.find_tables():
                try:
                    df = tbl.to_pandas()
                    if not is_real_table(df, text):
                        continue
                    tables.append({
                        "markdown": df.to_markdown(index=False),
                        "bbox": list(tbl.bbox),
                    })
                except Exception:
                    pass

            pages.append({
                "page_num": page_num + 1,
                "text":     text,
                "tables":   tables,
                # Embedded images aren't indexed; extracting them only cost
                # memory (25 MB for a 138-slide deck).
                "images":   [],
            })

        doc.close()
        return pages
