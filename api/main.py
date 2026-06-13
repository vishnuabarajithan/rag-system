from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router

app = FastAPI(
    title="Production RAG System",
    description="LangGraph-powered RAG with MLflow monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(ingest_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
