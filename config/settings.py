from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": stale keys left in an older .env (e.g. RERANKER_MODEL)
    # must not crash startup.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (query analysis + answer generation): Qwen 3.8 27B.
    # llm_provider "groq" (free tier, GROQ_API_KEY) or "hf" (Hugging Face
    # Inference Providers, HF_TOKEN; paid credits). llm_model empty = the
    # provider's default Qwen id (see agents/llm.py).
    llm_provider: str = "groq"
    llm_model: str = ""
    groq_api_key: str = ""
    hf_token: str = ""
    hf_provider: str = "auto"          # only for llm_provider == "hf"
    # Sized for Groq's free tier (~8K tokens/min): keep prompt + answer small.
    max_answer_tokens: int = 1024
    max_context_chars: int = 16_000    # ≈ 4K tokens of retrieved context

    # Google Gemini (image descriptions; embeddings only if embedding_provider == "gemini")
    google_api_key: str = ""
    vision_model: str = "gemini-2.5-flash"

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
    # "openrouter": qwen/qwen3-embedding-8b via OpenRouter (OPENROUTER_API_KEY;
    #   ~$0.01 per million tokens). "gemini": Gemini API (free tier: 100
    #   requests/min). "local": all-mpnet-base-v2 on CPU (requirements-local.txt).
    embedding_provider: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"
    embed_concurrency: int = 4          # parallel embedding requests during ingest
    embedding_model: str = "all-mpnet-base-v2"          # used when provider == "local"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # Retrieval: top_k hits per sub-query; the best context_top_n (after
    # cross-query fusion) go into the prompt.
    retriever_top_k: int = 20
    context_top_n: int = 6

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

    # Public-demo guard: caps on LLM-backed questions (0 = no cap).
    max_questions_per_session: int = 20
    # ~2 LLM calls per question; Groq's free tier allows ~200K tokens/day.
    daily_question_limit: int = 80


settings = Settings()

# Propagate an HF token (if provided) so huggingface_hub (and the optional local
# sentence-transformers embedder) make authenticated requests. No-op when unset.
if settings.hf_token:
    import os

    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)
