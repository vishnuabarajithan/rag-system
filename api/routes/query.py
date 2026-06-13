import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.rag_agent import rag_graph
from cache.semantic_cache import SemanticCache
from monitoring.mlflow_tracer import RAGTracer

router = APIRouter(prefix="/query", tags=["query"])
_cache = SemanticCache()


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    collections: list[str] = ["default"]


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence_score: float
    query_intent: str
    session_id: str
    mlflow_run_id: str | None = None


@router.post("/", response_model=QueryResponse)
async def query(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    tracer = RAGTracer(query=req.question, session_id=session_id)

    with tracer.start_run():
        # Semantic cache check
        cached = _cache.get(req.question)
        if cached:
            tracer.log_cache_hit()
            return QueryResponse(session_id=session_id, **cached)

        try:
            initial_state = {
                "user_query":       req.question,
                "session_id":       session_id,
                "collection_names": req.collections,
                "iteration_count":  0,
                "messages":         [],
            }

            with tracer.stage("graph_invoke"):
                result = rag_graph.invoke(initial_state)

            # Log individual stage metrics
            tracer.log_query_analysis(result)
            tracer.log_retrieval(result.get("raw_chunks", []), result.get("retrieval_quality", ""))
            tracer.log_reranking(result.get("reranked_chunks", []))
            tracer.log_generation(
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                confidence=result.get("confidence_score", 0.0),
            )

            answer      = result.get("final_answer", "")
            citations   = result.get("citations", [])
            confidence  = result.get("confidence_score", 0.0)
            query_intent= result.get("query_intent", "factual")

            tracer.log_final_answer(answer, citations)

            response_payload = {
                "answer":           answer,
                "citations":        citations,
                "confidence_score": confidence,
                "query_intent":     query_intent,
            }
            _cache.set(req.question, response_payload)

            return QueryResponse(
                session_id=session_id,
                mlflow_run_id=tracer.run_id,
                **response_payload,
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
