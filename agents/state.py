from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
import operator


class RetrievedChunk(TypedDict):
    chunk_id: str
    document_id: str
    content: str
    content_type: str           # "text" | "table" | "image_caption"
    score: float                # initial hybrid retrieval score
    metadata: dict              # filename, page_num, sheet_name, etc.


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    user_query: str
    session_id: str
    collection_names: list[str]
    source_filenames: list[str]   # restrict retrieval to these source files (optional)

    # ── Query understanding ─────────────────────────────────────────────────
    expanded_queries: list[str]
    query_intent: str           # "factual" | "analytical" | "summarize" | "compare"
    query_filters: dict         # metadata filters (file_type, date range, etc.)
    requires_table_reasoning: bool
    requires_image_understanding: bool

    # ── Retrieval ───────────────────────────────────────────────────────────
    raw_chunks: list[RetrievedChunk]
    top_chunks: list[RetrievedChunk]     # best context_top_n of raw_chunks, fused order
    context_window: str

    # ── Generation ──────────────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]
    final_answer: str
    citations: list[dict]       # [{ref, chunk_id, source_file, page_num}]
    confidence_score: float
    input_tokens: int
    output_tokens: int

    # ── Control flow ────────────────────────────────────────────────────────
    retrieval_quality: str      # "sufficient" | "insufficient" | "empty"
    iteration_count: int
    error: Optional[str]
