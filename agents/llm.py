"""
Shared LLM client for query analysis + answer generation.

Providers (settings.llm_provider), both OpenAI-compatible chat APIs driven
through huggingface_hub's InferenceClient:
  - "groq": Groq API (free tier), model `qwen/qwen3.8-27b`, key GROQ_API_KEY
  - "hf":   Hugging Face Inference Providers, model `Qwen/Qwen3.8-27B`, key HF_TOKEN

The client is lazy-initialized so importing this module (and the nodes that use
it) never needs a key — unit tests run without one.

Qwen3 models may emit a <think>…</think> reasoning block. We ask the provider to
disable thinking, and also strip any block that slips through, in both the
full-text and streaming paths.
"""
import json
import re
from typing import Iterator

from huggingface_hub import InferenceClient

from config.settings import settings

_client = None

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODELS = {"groq": "qwen/qwen3.8-27b", "hf": "Qwen/Qwen3.8-27B"}
# How each provider is told to skip Qwen's thinking phase.
_NO_THINK = {
    "groq": {"reasoning_effort": "none"},
    "hf":   {"chat_template_kwargs": {"enable_thinking": False}},
}


def _provider() -> str:
    return settings.llm_provider if settings.llm_provider in _DEFAULT_MODELS else "groq"


def model_name() -> str:
    return settings.llm_model or _DEFAULT_MODELS[_provider()]


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        if _provider() == "groq":
            _client = InferenceClient(
                base_url=_GROQ_BASE_URL, api_key=settings.groq_api_key or None
            )
        else:
            _client = InferenceClient(
                api_key=settings.hf_token or None,
                provider=settings.hf_provider or "auto",
            )
    return _client


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def strip_think(text: str) -> str:
    """Remove complete <think>…</think> blocks (and an unterminated leading one)."""
    text = _THINK_RE.sub("", text or "")
    if text.lstrip().startswith("<think>"):
        return ""
    return text.strip()


def filter_think(deltas: Iterator[str]) -> Iterator[str]:
    """Stream filter: if the output opens with <think>, drop everything up to
    and including </think>; otherwise pass deltas straight through."""
    buf = ""
    state = "detect"  # detect → (thinking →) passthrough
    for d in deltas:
        if state == "passthrough":
            yield d
            continue
        buf += d
        if state == "detect":
            head = buf.lstrip()
            if not head:
                continue
            if head.startswith("<think>"):
                state = "thinking"
            elif "<think>".startswith(head):
                continue  # could still become "<think>", wait for more
            else:
                state = "passthrough"
                yield buf
                buf = ""
                continue
        if state == "thinking" and "</think>" in buf:
            rest = buf.split("</think>", 1)[1].lstrip()
            state = "passthrough"
            buf = ""
            if rest:
                yield rest
    if state == "detect" and buf.strip():
        yield buf


def parse_json(text: str) -> dict:
    """Best-effort JSON object extraction; {} on failure."""
    text = _FENCE_RE.sub("", strip_think(text or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _tok(usage, key: str) -> int:
    """Read a token count from a usage object or plain dict."""
    if isinstance(usage, dict):
        return usage.get(key) or 0
    return getattr(usage, key, 0) or 0


def chat(system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
    """One-shot completion → (text, input_tokens, output_tokens)."""
    resp = _get_client().chat_completion(
        model=model_name(),
        messages=_messages(system, user),
        max_tokens=max_tokens,
        extra_body=_NO_THINK[_provider()],
    )
    text = resp.choices[0].message.content if resp.choices else ""
    usage = getattr(resp, "usage", None)
    in_tok = _tok(usage, "prompt_tokens") if usage else 0
    out_tok = _tok(usage, "completion_tokens") if usage else 0
    return strip_think(text or ""), in_tok, out_tok


def chat_stream(system: str, user: str, max_tokens: int, usage: dict) -> Iterator[str]:
    """Stream text deltas. `usage` is filled with input/output token counts
    from the final usage chunk, when the provider sends one."""
    stream = _get_client().chat_completion(
        model=model_name(),
        messages=_messages(system, user),
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=_NO_THINK[_provider()],
    )

    def deltas() -> Iterator[str]:
        for chunk in stream:
            # Standard `usage` (stream_options), or Groq's `x_groq.usage`.
            u = getattr(chunk, "usage", None) or (chunk.get("x_groq") or {}).get("usage")
            if u:
                usage["input_tokens"] = _tok(u, "prompt_tokens") or usage.get("input_tokens", 0)
                usage["output_tokens"] = _tok(u, "completion_tokens") or usage.get("output_tokens", 0)
            if not chunk.choices:
                continue
            text = getattr(chunk.choices[0].delta, "content", None)
            if text:
                yield text

    yield from filter_think(deltas())
