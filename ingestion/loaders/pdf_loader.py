from pathlib import Path
import fitz  # pymupdf


class PDFLoader:
    """
    Extracts text, tables (as markdown), and embedded images from a PDF.
    Returns one dict per page.
    """

    def load(self, path: Path) -> list[dict]:
        doc = fitz.open(str(path))
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()

            tables = []
            for tbl in page.find_tables():
                try:
                    df = tbl.to_pandas()
                    tables.append({
                        "markdown": df.to_markdown(index=False),
                        "bbox": list(tbl.bbox),
                    })
                except Exception:
                    pass

            images = []
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    images.append({
                        "bytes": base_image["image"],
                        "ext":   base_image["ext"],
                    })
                except Exception:
                    pass

            pages.append({
                "page_num": page_num + 1,
                "text":     text,
                "tables":   tables,
                "images":   images,
            })

        doc.close()
        return pages
