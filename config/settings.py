from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Google Gemini
    google_api_key: str = ""
    generation_model: str = "gemini-2.5-flash"
    routing_model: str = "gemini-2.5-flash"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_documents"
    # If set, run Qdrant in-process with no server (no Docker needed).
    # Use ":memory:" for ephemeral, or a folder path to persist across runs.
    qdrant_local_path: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "rag-system"

    # Embeddings
    embedding_model: str = "all-mpnet-base-v2"
    embedding_dim: int = 768

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 6
    retriever_top_k: int = 20

    # Cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4


settings = Settings()
