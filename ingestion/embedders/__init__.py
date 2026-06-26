from ingestion.embedders.text_embedder import TextEmbedder
from ingestion.embedders.image_embedder import ImageEmbedder
from config.settings import settings

__all__ = ["TextEmbedder", "ImageEmbedder", "get_embedder"]

# Process-wide text embedder singleton. Both the ingestion pipeline and the
# retriever node share this one instance so the model (or API client) is built
# once per process, not twice.
_embedder = None


def get_embedder():
    """Return the shared text embedder, chosen by settings.embedding_provider."""
    global _embedder
    if _embedder is None:
        if settings.embedding_provider == "local":
            _embedder = TextEmbedder()
        else:
            from ingestion.embedders.gemini_embedder import GeminiEmbedder
            _embedder = GeminiEmbedder()
    return _embedder
