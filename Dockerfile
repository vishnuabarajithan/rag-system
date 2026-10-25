# Hugging Face Spaces (Docker SDK) image for the web UI.
# No torch / local models: the LLM (Qwen via Groq) and embeddings (Qwen3
# Embedding via OpenRouter) are API calls; Qdrant runs embedded in-process.
FROM python:3.11-slim

# Tesseract for image OCR (ingestion/loaders/image_loader.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; the app dir must be writable by it
# (embedded Qdrant data, MLflow SQLite, FastHTML session key).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

ENV QDRANT_LOCAL_PATH=/home/user/app/qdrant_local \
    MLFLOW_ENABLED=false

EXPOSE 7860
# One worker: embedded Qdrant holds an exclusive file lock, and the
# in-memory session store + demo question counters are per-process.
CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
