"""
BM25 sparse vector encoder.
Converts text into (indices, values) pairs suitable for Qdrant sparse vectors.
A simple vocabulary-free approach: hash each token to a bucket index.
For production, replace with a fitted BM25Encoder from qdrant_client.
"""
import hashlib
import re
from collections import Counter


_MAX_VOCAB = 30_000  # bucket count for token hashing


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _bucket(token: str) -> int:
    # Stable across processes (unlike builtin hash, which is salted per run),
    # so vectors written at ingest time match those built at query time.
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _MAX_VOCAB


def bm25_encode(text: str, k1: float = 1.5, b: float = 0.75) -> tuple[list[int], list[float]]:
    """
    Returns (indices, values) for a sparse BM25 vector.
    Uses term frequency with IDF approximation via hash bucketing.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [], []

    tf = Counter(tokens)
    doc_len = len(tokens)
    avg_doc_len = 100  # assume average; calibrate per corpus in production

    # Aggregate by bucket so indices stay unique even when two tokens collide
    # (Qdrant rejects sparse vectors with duplicate indices).
    buckets: dict[int, float] = {}
    for token, freq in tf.items():
        bucket = _bucket(token)
        # BM25 TF component (no corpus-level IDF without an index)
        tf_score = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_doc_len))
        buckets[bucket] = buckets.get(bucket, 0.0) + tf_score

    return list(buckets.keys()), list(buckets.values())
