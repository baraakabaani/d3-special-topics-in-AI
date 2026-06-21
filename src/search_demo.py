"""
search_demo.py — D2.
done by Salim

Runs 5 diverse demo queries against /search and prints top-3 results
per query in full citation format for the README and notebook.

Usage:
  python -m src.search_demo
"""

import requests

API_URL = "http://localhost:8000/search"

DEMO_QUERIES = [
    "retrieval augmented generation architecture",
    "attention mechanism in transformer models",
    "fine-tuning large language models on domain data",
    "evaluation metrics for information retrieval",
    "vector similarity search and approximate nearest neighbours",
]


def search(query: str, top_k: int = 3) -> list[dict]:
    resp = requests.post(API_URL, json={"query": query, "top_k": top_k})
    resp.raise_for_status()
    return resp.json()


def format_result(rank: int, hit: dict) -> str:
    return (
        f"  [{rank}] {hit['title']}\n"
        f"      doc_id={hit['doc_id']}  page={hit['page']}  score={hit['score']:.4f}\n"
    )


def main():
    print("=" * 70)
    print("D2 Hybrid Search Demo  (BM25 + BGE-small via RRF)")
    print("=" * 70)

    for query in DEMO_QUERIES:
        print(f"\nQuery: {query!r}")
        print("-" * 70)
        try:
            results = search(query, top_k=3)
            if not results:
                print("  (no results returned)")
                continue
            for rank, hit in enumerate(results, start=1):
                print(format_result(rank, hit))
        except requests.HTTPError as e:
            print(f"  HTTP error: {e.response.status_code} {e.response.text}")
        except requests.ConnectionError:
            print("  Could not connect -- is the API running on localhost:8000?")

    print("=" * 70)


if __name__ == "__main__":
    main()
