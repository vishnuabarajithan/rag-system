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
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.entities import SpanType

from config.settings import settings

logger = logging.getLogger(__name__)

# Configuration is done lazily (on first run), never at import time, so that
# importing this module never touches the network / tracking backend.
_configured: bool | None = None


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _ensure_configured() -> bool:
    """Configure MLflow once; return True if monitoring is usable.

    Idempotent and guarded: a misconfigured / unreachable backend disables
    monitoring rather than raising, so a query is never blocked by it.
    """
    global _configured
    if _configured is not None:
        return _configured
    if not settings.mlflow_enabled:
        _configured = False
        return False
    try:
        configure_mlflow()
        _configured = True
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("MLflow disabled (configuration failed): %s", exc)
        _configured = False
    return _configured


def _safe(fn) -> None:
    """Run an MLflow logging call, swallowing any failure (monitoring must
    never break a query)."""
    try:
        fn()
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow logging call failed: %s", exc)


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
        _ensure_configured()
        with mlflow.start_run(run_name=f"query-{self.session_id[:8]}") as run:
            self._run = run
            _safe(lambda: mlflow.log_params({
                "session_id":       self.session_id,
                "llm_provider":     settings.llm_provider,
                "llm_model":        settings.llm_model or "(provider default)",
                "retriever_top_k":  settings.retriever_top_k,
                "context_top_n":    settings.context_top_n,
                "embedding_model":  settings.embedding_model,
            }))
            _safe(lambda: mlflow.log_param("query_preview", self.query[:200]))
            try:
                yield self
            except Exception as exc:
                # Record the failure, but never let a logging error mask the
                # original exception — re-raise so the caller can handle it.
                _safe(lambda: mlflow.log_param("error", str(exc)[:500]))
                _safe(lambda: mlflow.set_tag("status", "failed"))
                raise
            finally:
                total_ms = (time.perf_counter() - self._run_start) * 1000
                _safe(lambda: mlflow.log_metric("total_latency_ms", total_ms))

    @contextmanager
    def stage(self, name: str):
        """Time a pipeline stage and log its latency as a metric."""
        timer = StageTimer(name)
        with mlflow.start_span(name=name, span_type=SpanType.CHAIN) as span:
            _safe(lambda: span.set_attribute("stage", name))
            yield timer
            elapsed = timer.elapsed_ms()
            _safe(lambda: mlflow.log_metric(f"{name}_latency_ms", elapsed))
            _safe(lambda: span.set_attribute("latency_ms", elapsed))

    def log_query_analysis(self, analysis: dict) -> None:
        _safe(lambda: mlflow.log_params({
            "query_intent":                 analysis.get("query_intent", ""),
            "num_expanded_queries":         len(analysis.get("expanded_queries", [])),
            "requires_table_reasoning":     str(analysis.get("requires_table_reasoning", False)),
            "requires_image_understanding": str(analysis.get("requires_image_understanding", False)),
        }))

    def log_retrieval(self, raw_chunks: list[dict], quality: str) -> None:
        _safe(lambda: mlflow.log_metric("num_raw_chunks", len(raw_chunks)))
        _safe(lambda: mlflow.log_param("retrieval_quality", quality))

        if raw_chunks:
            scores = [c.get("score", 0.0) for c in raw_chunks]
            _safe(lambda: mlflow.log_metrics({
                "retrieval_max_score":  max(scores),
                "retrieval_mean_score": sum(scores) / len(scores),
            }))

    def log_generation(
        self,
        input_tokens: int,
        output_tokens: int,
        confidence: float,
    ) -> None:
        _safe(lambda: mlflow.log_metrics({
            "input_tokens":     input_tokens,
            "output_tokens":    output_tokens,
            "total_tokens":     input_tokens + output_tokens,
            "confidence_score": confidence,
        }))

    def log_final_answer(self, answer: str, citations: list[dict]) -> None:
        """Save full answer text and citations as MLflow artifacts."""
        def _log_artifacts():
            with tempfile.TemporaryDirectory() as tmp:
                answer_path = os.path.join(tmp, "answer.txt")
                citations_path = os.path.join(tmp, "citations.json")

                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(answer)
                with open(citations_path, "w", encoding="utf-8") as f:
                    json.dump(citations, f, indent=2)

                mlflow.log_artifact(answer_path)
                mlflow.log_artifact(citations_path)

        _safe(_log_artifacts)
        _safe(lambda: mlflow.set_tag("status", "success"))
        _safe(lambda: mlflow.log_metric("num_citations", len(citations)))

    def log_cache_hit(self) -> None:
        _safe(lambda: mlflow.set_tag("cache_hit", "true"))
        _safe(lambda: mlflow.log_metric(
            "total_latency_ms", (time.perf_counter() - self._run_start) * 1000))

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None


class NoopTracer:
    """Drop-in stand-in used when monitoring is disabled or unconfigurable.

    Mirrors RAGTracer's interface; every method is a no-op so callers don't
    need to branch on whether monitoring is active.
    """

    def __init__(self, *args, **kwargs):
        pass

    @contextmanager
    def start_run(self):
        yield self

    @contextmanager
    def stage(self, name: str):
        yield StageTimer(name)

    def log_query_analysis(self, *a, **k) -> None: ...
    def log_retrieval(self, *a, **k) -> None: ...
    def log_generation(self, *a, **k) -> None: ...
    def log_final_answer(self, *a, **k) -> None: ...
    def log_cache_hit(self, *a, **k) -> None: ...

    @property
    def run_id(self) -> None:
        return None


def get_tracer(query: str, session_id: str) -> "RAGTracer | NoopTracer":
    """Return a real tracer when MLflow monitoring is enabled and usable,
    otherwise a NoopTracer. Callers use the same interface either way."""
    if _ensure_configured():
        return RAGTracer(query=query, session_id=session_id)
    return NoopTracer()
