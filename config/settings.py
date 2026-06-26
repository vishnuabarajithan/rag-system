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
    # Local default: a SQLite file in the project root — works with no server
    # and no Docker. View it with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    # In Docker, this is overridden to the MLflow server (http://mlflow:5000).
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "rag-system"
    mlflow_enabled: bool = True

    # Embeddings
    # "gemini" uses the Gemini API (no local model, no GPU); "local" uses a
    # SentenceTransformer (all-mpnet-base-v2) on CPU.
    embedding_provider: str = "gemini"
    embedding_model: str = "all-mpnet-base-v2"          # used when provider == "local"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # HuggingFace (optional): set to silence the "unauthenticated requests to the
    # HF Hub" warning and get higher rate limits / faster downloads.
    hf_token: str = ""

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 6
    retriever_top_k: int = 20

    # Cache
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Web UI (FastHTML) sessions
    # SESSION_SECRET signs the session cookie — set a real secret in production.
    # SESSION_BACKEND: "memory" (single process / local) or "redis" (multi-worker).
    session_secret: str = "dev-insecure-change-me"
    session_backend: str = "memory"


settings = Settings()

# Propagate an HF token (if provided) so sentence-transformers / huggingface_hub
# make authenticated requests — silences the "unauthenticated requests to the HF
# Hub" warning and lifts download rate limits. No-op when unset.
if settings.hf_token:
    import os

    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)
