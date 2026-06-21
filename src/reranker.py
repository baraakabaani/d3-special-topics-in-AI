"""
reranker.py — D3.
done by Yousef

CrossEncoder reranker wrapper around cross-encoder/ms-marco-MiniLM-L-6-v2.
Loaded once as a module-level singleton; subsequent calls skip model init.

Usage:
  from src.reranker import rerank
  top_chunks = rerank(query, chunks, top_k=5)
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score each chunk against the query using a CrossEncoder and return the
    top_k chunks sorted by relevance score descending.

    Each chunk dict must have a "text" field.
    The reranker score is added as "rerank_score" on returned chunks.
    """
    if not chunks:
        return []

    model  = _get_model()
    pairs  = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
