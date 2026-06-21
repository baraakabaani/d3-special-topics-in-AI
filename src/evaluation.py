"""
Evaluation: NDCG@5, Recall@5, semantic similarity, p95 latency, prequential chart.
done by Salim

D2 addition: semantic_similarity_at_k() measures TF-IDF cosine similarity between
the query and each retrieved document — a semantic layer on top of IR metrics.
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ndcg_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    rel_set = set(relevant)
    dcg  = sum(1 / np.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in rel_set)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    rel_set = set(relevant)
    return len(set(retrieved[:k]) & rel_set) / len(rel_set) if rel_set else 0.0


def semantic_similarity_at_k(query: str, retrieved_ids: list,
                              corpus: list, k: int = 5) -> float:
    """
    Mean TF-IDF cosine similarity between the query and the top-k retrieved docs.
    Measures semantic overlap beyond exact-match IR metrics.
    Returns a score in [0, 1].
    """
    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return 0.0
    docs = [corpus[i] for i in top_k_ids if i < len(corpus)]
    if not docs:
        return 0.0
    vect = TfidfVectorizer().fit(docs + [query])
    doc_vecs   = vect.transform(docs)
    query_vec  = vect.transform([query])
    sims = cosine_similarity(query_vec, doc_vecs)[0]
    return float(np.mean(sims))


def evaluate_semantic(queries: list, retriever, corpus: list, k: int = 5) -> float:
    """Average semantic_similarity_at_k over all queries."""
    scores = [
        semantic_similarity_at_k(q, retriever.retrieve(q, k=k), corpus, k)
        for q in queries
    ]
    return float(np.mean(scores))


def measure_p95_latency(retriever, queries: list, n: int = 200) -> float:
    sample = (queries * ((n // len(queries)) + 1))[:n]
    lats = []
    for q in sample:
        t0 = time.perf_counter()
        retriever.retrieve(q)
        lats.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(lats, 95))


def plot_prequential(log_path: str, out_path: str) -> None:
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)

    values       = log["values"]
    windowed     = log.get("windowed_values", [])
    drift_points = log.get("drift_points", [])
    win_size     = log.get("windowed_size", 50)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values,   linewidth=1.2, label="Cumulative accuracy",          color="#2166AC")
    if windowed:
        ax.plot(windowed, linewidth=1.0, alpha=0.8,
                label=f"Sliding window (n={win_size})", color="#4DAC26")
    for i, dp in enumerate(drift_points):
        ax.axvline(dp, color="#D85A30", linestyle="--", linewidth=0.9,
                   label="ADWIN drift" if i == 0 else "")

    ax.set_xlabel("Feedback event index")
    ax.set_ylabel("Accuracy")
    ax.set_title("River Online Learner - Prequential Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved prequential chart -> {out_path}")
