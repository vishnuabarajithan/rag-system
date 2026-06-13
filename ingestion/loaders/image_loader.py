"""
Image loader: runs OCR (pytesseract) for text extraction and calls
Gemini for a rich visual description. Both are combined into a
single text chunk for embedding.
"""
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
import pytesseract

from config.settings import settings

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client

_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


class ImageLoader:
    def load(self, path: Path) -> dict:
        image = Image.open(path)

        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(image).strip()
        except Exception:
            pass

        caption = self._caption(path)

        combined = ""
        if ocr_text:
            combined += f"[OCR TEXT]: {ocr_text}\n\n"
        combined += f"[VISUAL DESCRIPTION]: {caption}"

        return {
            "ocr_text": ocr_text,
            "caption":  caption,
            "combined": combined,
            "path":     str(path),
        }

    def _caption(self, path: Path) -> str:
        ext = path.suffix.lstrip(".").lower()
        media_type = _MEDIA_TYPES.get(ext, "image/png")

        response = _get_client().models.generate_content(
            model=settings.routing_model,
            contents=[
                types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=media_type,
                ),
                (
                    "Describe this image in detail, focusing on all text, data, "
                    "charts, diagrams, or figures visible. Include any numbers, "
                    "labels, or legends."
                ),
            ],
            config=types.GenerateContentConfig(max_output_tokens=512),
        )
        return response.text or ""
