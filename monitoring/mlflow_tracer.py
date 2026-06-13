"""
MLflow monitoring for the RAG system.

Each query becomes an MLflow run under the experiment defined in settings.
The RAGTracer context manager handles run lifecycle, logging params/metrics/
artifacts and any child spans for individual pipeline stages.

MLflow 2.14+ tracing API is used for span-level observability.
"""
import time
import json
import tempfile
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities import SpanType

from config.settings import settings


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


configure_mlflow()


@dataclass
class StageTimer:
    name: str
    start: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000


class RAGTracer:
    """
    Wraps a single RAG query in an MLflow run and exposes helpers to log
    each pipeline stage as a child span.

    Usage:
        tracer = RAGTracer(query="...", session_id="...")
        with tracer.start_run():
            with tracer.stage("retrieval") as t:
                chunks = retriever.run(...)
                tracer.log_retrieval(chunks)
            tracer.log_final(state)
    """

    def __init__(self, query: str, session_id: str):
        self.query = query
        self.session_id = session_id
        self._run: mlflow.ActiveRun | None = None
        self._run_start = 0.0

    @contextmanager
    def start_run(self):
        self._run_start = time.perf_counter()
        with mlflow.start_run(run_name=f"query-{self.session_id[:8]}") as run:
            self._run = run
            mlflow.log_params({
                "session_id":       self.session_id,
                "generation_model": settings.generation_model,
                "routing_model":    settings.routing_model,
                "retriever_top_k":  settings.retriever_top_k,
                "reranker_top_n":   settings.reranker_top_n,
                "embedding_model":  settings.embedding_model,
                "reranker_model":   settings.reranker_model,
            })
            mlflow.log_param("query_preview", self.query[:200])
            try:
                yield self
            except Exception as exc:
                mlflow.log_param("error", str(exc)[:500])
                mlflow.set_tag("status", "failed")
                raise
            finally:
                total_ms = (time.perf_counter() - self._run_start) * 1000
                mlflow.log_metric("total_latency_ms", total_ms)

    @contextmanager
    def stage(self, name: str):
        """Time a pipeline stage and log its latency as a metric."""
        timer = StageTimer(name)
        with mlflow.start_span(name=name, span_type=SpanType.CHAIN) as span:
            span.set_attribute("stage", name)
            yield timer
            elapsed = timer.elapsed_ms()
            mlflow.log_metric(f"{name}_latency_ms", elapsed)
            span.set_attribute("latency_ms", elapsed)

    def log_query_analysis(self, analysis: dict) -> None:
        mlflow.log_params({
            "query_intent":                 analysis.get("query_intent", ""),
            "num_expanded_queries":         len(analysis.get("expanded_queries", [])),
            "requires_table_reasoning":     str(analysis.get("requires_table_reasoning", False)),
            "requires_image_understanding": str(analysis.get("requires_image_understanding", False)),
        })

    def log_retrieval(self, raw_chunks: list[dict], quality: str) -> None:
        mlflow.log_metrics({
            "num_raw_chunks":   len(raw_chunks),
        })
        mlflow.log_param("retrieval_quality", quality)

        if raw_chunks:
            scores = [c.get("score", 0.0) for c in raw_chunks]
            mlflow.log_metrics({
                "retrieval_max_score":  max(scores),
                "retrieval_mean_score": sum(scores) / len(scores),
            })

    def log_reranking(self, reranked_chunks: list[dict]) -> None:
        mlflow.log_metric("num_reranked_chunks", len(reranked_chunks))
        if reranked_chunks:
            scores = [c.get("rerank_score", 0.0) for c in reranked_chunks]
            mlflow.log_metrics({
                "rerank_top_score":  max(scores),
                "rerank_mean_score": sum(scores) / len(scores),
            })

    def log_generation(
        self,
        input_tokens: int,
        output_tokens: int,
        confidence: float,
    ) -> None:
        mlflow.log_metrics({
            "input_tokens":     input_tokens,
            "output_tokens":    output_tokens,
            "total_tokens":     input_tokens + output_tokens,
            "confidence_score": confidence,
        })

    def log_final_answer(self, answer: str, citations: list[dict]) -> None:
        """Save full answer text and citations as MLflow artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            answer_path = os.path.join(tmp, "answer.txt")
            citations_path = os.path.join(tmp, "citations.json")

            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(answer)
            with open(citations_path, "w", encoding="utf-8") as f:
                json.dump(citations, f, indent=2)

            mlflow.log_artifact(answer_path)
            mlflow.log_artifact(citations_path)

        mlflow.set_tag("status", "success")
        mlflow.log_metric("num_citations", len(citations))

    def log_cache_hit(self) -> None:
        mlflow.set_tag("cache_hit", "true")
        mlflow.log_metric("total_latency_ms",
                          (time.perf_counter() - self._run_start) * 1000)

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None
