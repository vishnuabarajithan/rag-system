from pathlib import Path


class TextLoader:
    """Loads plain text and markdown files."""

    SUPPORTED = {".txt", ".md", ".rst", ".log", ".csv"}

    def load(self, path: Path) -> list[dict]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [{"page_num": 1, "text": text, "tables": [], "images": []}]
