"""
eval_metrics.py — D3.
done by Faisal

Implements all retrieval and answer-quality metrics from scratch.
No external evaluation libraries — every formula matches the professor's
lab notebook implementations.

Retrieval metrics (operate on ranked doc_id lists):
  recall_at_k      — fraction of relevant docs appearing in top-k
  precision_at_k   — fraction of top-k results that are relevant
  mrr              — reciprocal rank of first relevant result
  ndcg_at_k        — position-discounted gain, normalised by ideal ranking

Answer quality metrics (operate on generated answer text):
  faithfulness_score    — fraction of answer sentences supported by context
  answer_relevance_score — cosine similarity between question and answer embeddings
"""

import math
import re
from collections import Counter


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Fraction of relevant documents that appear in the top-k retrieved list.

    recall@k = |relevant ∩ top-k| / |relevant|

    Returns 0.0 when the relevant set is empty (undefined recall is treated
    as 0 rather than 1 to avoid inflating aggregate scores).
    """
    if not relevant_ids:
        return 0.0
    top_k   = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Fraction of the top-k retrieved results that are relevant.

    precision@k = |relevant ∩ top-k| / k
    """
    if k == 0:
        return 0.0
    top_k    = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits     = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def mrr(retrieved_ids: list[str], relevant_ids: list[str], max_k: int = 10) -> float:
    """
    Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.

    MRR = 1 / rank_of_first_relevant   (0 if no relevant in top max_k)

    max_k caps the search window so results beyond max_k do not contribute.
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:max_k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k (binary relevance).

    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)   where rel_i ∈ {0, 1}
    IDCG@k = sum_{i=1}^{min(|relevant|, k)} 1 / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

    IDCG uses the actual relevant count so questions with two expected docs
    are not penalized by a fixed denominator designed for single-doc questions.
    """
    relevant  = set(relevant_ids)
    n_relevant = min(len(relevant), k)

    if n_relevant == 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)           # rank is 0-indexed; log2(1+1)=1 at rank 0
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant
    )

    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


# ── Answer quality metrics ────────────────────────────────────────────────────

def _tokenize(text: str) -> Counter:
    """Lowercase alphabetic tokens of length >= 3 as a frequency Counter."""
    return Counter(re.findall(r"[a-z]{3,}", text.lower()))


def faithfulness_score(answer: str, context_chunks: list[dict]) -> float:
    """
    Fraction of answer sentences whose content tokens are supported by
    the retrieved context.

    Algorithm:
      1. Split answer on sentence-ending punctuation.
      2. Skip sentences with fewer than 5 content words (too short to evaluate).
      3. For each remaining sentence compute token overlap with the union of
         all context chunk texts.
      4. A sentence is "supported" if overlap ratio >= 0.30 (30% of its
         tokens appear in context).
      5. Score = supported_sentences / total_evaluated_sentences.

    Edge cases:
      - If the answer is a refusal ("I cannot answer from the provided context")
        all evaluated sentences will overlap with nothing and score 0.0, which
        is correct — a refusal is not faithful to the context, it is absent.
      - If no evaluable sentences exist, returns 1.0 (vacuously faithful).

    The 0.30 threshold is deliberately lenient to account for paraphrasing;
    stricter thresholds (0.50) are appropriate only with exact-extraction RAG.
    """
    if not answer or not context_chunks:
        return 0.0

    context_text   = " ".join(c.get("text", "") for c in context_chunks)
    context_tokens = _tokenize(context_text)

    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    evaluable = [s for s in sentences if len(s.split()) >= 5]

    if not evaluable:
        return 1.0

    supported = 0
    for sentence in evaluable:
        sent_tokens = _tokenize(sentence)
        if not sent_tokens:
            continue
        overlap = sum(
            min(sent_tokens[t], context_tokens[t]) for t in sent_tokens
        )
        overlap_ratio = overlap / sum(sent_tokens.values())
        if overlap_ratio >= 0.30:
            supported += 1

    return supported / len(evaluable)


def answer_relevance_score(question: str, answer: str, embedder) -> float:
    """
    Cosine similarity between the BGE-small embeddings of the question
    and the generated answer.

    Uses the same embedder singleton as the retrieval pipeline to avoid
    loading a second model. BGE vectors are L2-normalised, so cosine
    similarity equals the dot product.

    A high score means the answer addresses the question topically.
    A low score means the model went off-topic or returned a refusal.
    """
    import numpy as np

    q_emb = embedder.encode([question], normalize_embeddings=True)[0]
    a_emb = embedder.encode([answer],   normalize_embeddings=True)[0]
    return float(np.dot(q_emb, a_emb))
