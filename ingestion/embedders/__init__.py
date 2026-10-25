from config.settings import settings

__all__ = ["get_embedder"]

# Process-wide text embedder singleton. Both the ingestion pipeline and the
# retriever node share this one instance so the model (or API client) is built
# once per process, not twice.
#
# Embedder modules are imported lazily, so providers that aren't in use are
# never loaded — in particular the local one's torch / sentence-transformers
# (requirements-local.txt), which the default install doesn't ship.
_embedder = None


def get_embedder():
    """Return the shared text embedder, chosen by settings.embedding_provider."""
    global _embedder
    if _embedder is None:
        if settings.embedding_provider == "openrouter":
            from ingestion.embedders.openrouter_embedder import OpenRouterEmbedder
            _embedder = OpenRouterEmbedder()
        elif settings.embedding_provider == "local":
            from ingestion.embedders.text_embedder import TextEmbedder
            _embedder = TextEmbedder()
        else:
            from ingestion.embedders.gemini_embedder import GeminiEmbedder
            _embedder = GeminiEmbedder()
    return _embedder
