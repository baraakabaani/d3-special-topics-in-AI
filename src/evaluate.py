"""
evaluate.py — D2.
done by Faisal

Measures Recall@k and latency for the /search endpoint using
pseudo-relevance: each query is crafted from a known paper's title,
and a result is "relevant" if the expected doc_id appears in top-k.

Usage:
  python -m src.evaluate
"""

import os
import statistics
import time

import requests
from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# (query_text, expected_doc_id)
# Queries crafted from actual paper titles in our corpus
TEST_QUERIES = [
    ("knowledge graph traversal document injection",              "2604.27820v1"),
    ("retrieval augmented generation evidence explicit",          "2604.27852v1"),
    ("unified toolkit benchmark evaluating information retrieval","2604.27878v1"),
    ("text to sql accuracy recurring questions reliable answers", "2604.28028v1"),
    ("multivector retrieval token aware clustering hierarchical", "2604.28142v1"),
    ("LLM biases manipulate AI search overview",                  "2605.00012v1"),
    ("reasoning intensive retrieval survey progress challenges",  "2605.00063v1"),
    ("structured attribution small language models table reasoning","2605.00199v2"),
    ("retrieval augmented reasoning chartered accountancy",       "2605.00257v1"),
    ("structure aware chunking tabular data retrieval generation","2605.00318v1"),
]


def _search(query: str, top_k: int) -> tuple[list[dict], float]:
    t0   = time.perf_counter()
    resp = requests.post(
        f"{API_URL}/search",
        json={"query": query, "top_k": top_k},
        timeout=180,   # first request downloads bge-small model (~400MB cold start)
    )
    lat = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json(), lat


def evaluate(ks: list[int] = [1, 3, 5]) -> dict:
    print(f"Evaluating {len(TEST_QUERIES)} queries against {API_URL}/search\n")
    latencies: list[float] = []
    hits = {k: 0 for k in ks}

    for query, expected_doc in TEST_QUERIES:
        results, lat = _search(query, top_k=max(ks))
        latencies.append(lat)

        for k in ks:
            top_docs = [r["doc_id"] for r in results[:k]]
            if expected_doc in top_docs:
                hits[k] += 1

        top1 = results[0]["doc_id"] if results else "none"
        hit  = "HIT " if expected_doc in [r["doc_id"] for r in results[:5]] else "MISS"
        print(f"  {hit}  Q: {query[:50]:<50s}  lat={lat*1000:.0f}ms  top1={top1}")

    n      = len(TEST_QUERIES)
    recall = {k: round(hits[k] / n, 3) for k in ks}
    mean_lat = statistics.mean(latencies) * 1000
    std_lat  = statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0

    print(f"\n{'-'*50}")
    print(f"{'Metric':<25} {'Value':>10}")
    print(f"{'-'*50}")
    for k in ks:
        print(f"  Recall@{k:<18} {recall[k]:>10.3f}")
    print(f"  {'Mean latency (ms)':<23} {mean_lat:>10.1f}")
    print(f"  {'Std  latency (ms)':<23} {std_lat:>10.1f}")
    print(f"{'-'*50}")
    print("\nNote: BM25 index is rebuilt per HybridSearch() instantiation.")
    print("      Latency can be cached in D3 by keeping a singleton instance.")

    return {
        "recall":          recall,
        "mean_latency_ms": round(mean_lat, 1),
        "std_latency_ms":  round(std_lat, 1),
    }


if __name__ == "__main__":
    evaluate()
