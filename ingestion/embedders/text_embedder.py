from sentence_transformers import SentenceTransformer
from config.settings import settings


class TextEmbedder:
    def __init__(self, model: str | None = None):
        self._model = SentenceTransformer(model or settings.embedding_model)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()
