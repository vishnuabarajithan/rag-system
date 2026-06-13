from sentence_transformers import CrossEncoder as _CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        self._model = _CrossEncoder(model, max_length=512)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 6,
    ) -> list[tuple[float, int]]:
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs).tolist()

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(score, idx) for idx, score in ranked[:top_n]]
