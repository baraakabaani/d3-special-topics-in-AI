"""
run_eval.py — D3.
done by Faisal

Runs the ablation study across three retrieval modes and computes all
evaluation metrics defined in eval_metrics.py.

Modes
-----
  A  vector-only   — existing HybridSearch (BM25 + Qdrant ANN), no graph
  B  graph-guided  — Neo4j subgraph selection + chunk expansion + dense rerank,
                     no BM25, no CrossEncoder
  C  hybrid+rerank — full GraphRAGExecutor pipeline with CrossEncoder reranker
                     + Groq answer generation

Retrieval metrics (Modes A, B, C):
  Recall@1, Recall@3, Recall@5
  Precision@3
  MRR (max_k=10)
  NDCG@5

Answer quality metrics (Mode C only — Groq answer required):
  Faithfulness     — token overlap between answer and retrieved chunks
  Answer relevance — BGE cosine similarity between question and answer

Latency:
  Mean latency (ms) and p95 latency (ms) per mode

Usage:
  python -m src.run_eval
"""

import json
import os
import statistics
import time

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.eval_metrics import (
    answer_relevance_score,
    faithfulness_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

load_dotenv(".env.local", override=True)

API_URL   = os.getenv("API_URL", "http://localhost:8000")
GOLD_PATH = "data/gold_qa.json"
KS        = [1, 3, 5]


# ── data loading ──────────────────────────────────────────────────────────────

def load_gold() -> list[dict]:
    with open(GOLD_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── per-mode search calls ─────────────────────────────────────────────────────

def _call(url: str, payload: dict, timeout: int = 120) -> tuple[dict, float]:
    max_retries = 4
    backoff     = 2

    for attempt in range(max_retries):
        t0   = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=timeout)
        lat  = time.perf_counter() - t0

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json(), lat

        wait = int(resp.headers.get("Retry-After", backoff))
        if attempt < max_retries - 1:
            print(f"    [429] rate limited — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            backoff *= 2
        else:
            resp.raise_for_status()

    return resp.json(), lat


def search_mode_a(query: str, top_k: int = 5) -> tuple[list[dict], float]:
    """Vector-only: standard /search endpoint (BM25 + Qdrant RRF)."""
    data, lat = _call(f"{API_URL}/search", {"query": query, "top_k": top_k})
    return data, lat


def search_mode_b(query: str, top_k: int = 5) -> tuple[list[dict], float]:
    """Graph-guided: /search/graph without CrossEncoder reranker."""
    data, lat = _call(
        f"{API_URL}/search/graph",
        {"query": query, "top_k": top_k, "rerank": False},
    )
    # endpoint returns {"chunks": [...], ...}
    return data.get("chunks", data), lat


def search_mode_c(query: str, top_k: int = 5) -> tuple[list[dict], str, float]:
    """Hybrid+rerank: full /answer pipeline with CrossEncoder + Groq."""
    data, lat = _call(
        f"{API_URL}/answer",
        {"query": query, "top_k": top_k, "rerank": True},
        timeout=180,
    )
    return data.get("chunks", []), data.get("answer", ""), lat


# ── single-mode evaluation ────────────────────────────────────────────────────

def evaluate_mode(mode: str, gold: list[dict], embedder) -> dict:
    print(f"\n{'='*65}")
    print(f"  Mode {mode}")
    print('='*65)

    latencies:    list[float] = []
    all_recall    = {k: [] for k in KS}
    all_precision = {k: [] for k in KS}
    all_mrr:      list[float] = []
    all_ndcg:     list[float] = []
    all_faith:    list[float] = []
    all_relevance:list[float] = []

    for item in gold:
        query    = item["question"]
        relevant = item["expected_doc_ids"]

        if mode == "A":
            chunks, lat     = search_mode_a(query, top_k=max(KS))
            answer_text     = None

        elif mode == "B":
            chunks, lat     = search_mode_b(query, top_k=max(KS))
            answer_text     = None

        else:  # C
            chunks, answer_text, lat = search_mode_c(query, top_k=max(KS))
            if not chunks:
                print(f"    [WARN] /answer returned no chunks for: {query[:50]}")

        # deduplicate doc_ids while preserving rank order (multiple chunks per doc
        # would otherwise inflate precision and NDCG by counting the same doc twice)
        seen: set[str] = set()
        retrieved_ids: list[str] = []
        for c in chunks:
            did = c["doc_id"]
            if did not in seen:
                retrieved_ids.append(did)
                seen.add(did)

        latencies.append(lat)

        for k in KS:
            all_recall[k].append(recall_at_k(retrieved_ids, relevant, k))
            all_precision[k].append(precision_at_k(retrieved_ids, relevant, k))

        all_mrr.append(mrr(retrieved_ids, relevant, max_k=10))
        all_ndcg.append(ndcg_at_k(retrieved_ids, relevant, k=5))

        # Answer quality only when Groq answer is available (Mode C)
        if answer_text:
            all_faith.append(faithfulness_score(answer_text, chunks))
            all_relevance.append(answer_relevance_score(query, answer_text, embedder))

        if mode == "C":
            time.sleep(3)

        hit  = "HIT " if any(r in retrieved_ids[:3] for r in relevant) else "MISS"
        print(f"  {hit}  {query[:55]:<55}  {lat*1000:.0f}ms")

    # Latency p95: sort and take the 95th percentile sample
    lats_ms     = [l * 1000 for l in latencies]
    sorted_lats = sorted(lats_ms)
    p95_idx     = max(0, int(len(sorted_lats) * 0.95) - 1)
    p95         = sorted_lats[p95_idx] if sorted_lats else 0.0

    result = {
        "mode":            mode,
        "recall":          {k: round(statistics.mean(all_recall[k]), 3) for k in KS},
        "precision":       {k: round(statistics.mean(all_precision[k]), 3) for k in KS},
        "mrr":             round(statistics.mean(all_mrr), 3),
        "ndcg_at_5":       round(statistics.mean(all_ndcg), 3),
        "mean_latency_ms": round(statistics.mean(lats_ms), 1),
        "p95_latency_ms":  round(p95, 1),
    }

    if all_faith:
        result["faithfulness"]      = round(statistics.mean(all_faith), 3)
    if all_relevance:
        result["answer_relevance"]  = round(statistics.mean(all_relevance), 3)

    return result


# ── results table ─────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    print(f"\n{'='*75}")
    print("  ABLATION RESULTS")
    print('='*75)
    print(f"  {'Metric':<24} {'Mode A':>10} {'Mode B':>10} {'Mode C':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")

    rows = [
        ("Recall@1",        lambda r: r["recall"][1]),
        ("Recall@3",        lambda r: r["recall"][3]),
        ("Recall@5",        lambda r: r["recall"][5]),
        ("Precision@3",     lambda r: r["precision"][3]),
        ("MRR",             lambda r: r["mrr"]),
        ("NDCG@5",          lambda r: r["ndcg_at_5"]),
        ("Faithfulness",    lambda r: r.get("faithfulness",     "N/A")),
        ("Ans. Relevance",  lambda r: r.get("answer_relevance", "N/A")),
        ("Mean lat. (ms)",  lambda r: r["mean_latency_ms"]),
        ("p95  lat. (ms)",  lambda r: r["p95_latency_ms"]),
    ]

    for label, fn in rows:
        vals = [str(fn(r)) for r in results]
        print(f"  {label:<24} " + "".join(f"{v:>10}" for v in vals))

    print('='*75)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    gold     = load_gold()
    print(f"Loaded {len(gold)} gold Q/A pairs")

    all_results = []
    for mode in ["A", "B", "C"]:
        r = evaluate_mode(mode, gold, embedder)
        all_results.append(r)

    print_table(all_results)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
