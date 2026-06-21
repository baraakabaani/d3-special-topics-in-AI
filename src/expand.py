"""
expand.py — D3.
done by Salim

Two functions used by GraphRAGExecutor:

  expand_chunks()   — given doc_ids from the graph selector, fetch their
                      actual text chunks from MongoDB (d2.chunks).

  blend_results()   — RRF fusion of graph-expanded chunks and Qdrant vector
                      chunks into a single ranked list before CrossEncoder rerank.
"""

from pymongo.collection import Collection


def expand_chunks(
    doc_ids: list[str],
    collection: Collection,
    chunks_per_doc: int = 3,
) -> list[dict]:
    """
    Fetch up to `chunks_per_doc` chunks per doc_id from MongoDB, sorted by
    chunk_index ascending so we get the earliest (most context-rich) chunks.

    Returns list of dicts with at minimum: doc_id, text, chunk_index, page_start.
    """
    if not doc_ids:
        return []

    results = []
    for doc_id in doc_ids:
        chunks = list(
            collection.find(
                {"doc_id": doc_id},
                {"_id": 0, "doc_id": 1, "chunk_id": 1, "text": 1,
                 "chunk_index": 1, "page_start": 1, "page_end": 1, "title": 1},
            )
            .sort("chunk_index", 1)
            .limit(chunks_per_doc)
        )
        results.extend(chunks)
    return results


def blend_results(
    graph_chunks: list[dict],
    vector_chunks: list[dict],
    graph_weight: float = 0.4,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of graph-expanded chunks and vector chunks.

    RRF score per chunk = graph_weight / (60 + graph_rank)
                        + vector_weight / (60 + vector_rank)

    Chunks appearing in both lists accumulate from both terms.
    Deduplication key is (doc_id, chunk_index).
    Returns list sorted by blended RRF score descending.
    """
    vector_weight = 1.0 - graph_weight
    scores:    dict[tuple, float] = {}
    chunk_map: dict[tuple, dict]  = {}

    def _key(c: dict) -> tuple:
        return (c["doc_id"], c.get("chunk_index", 0))

    for rank, chunk in enumerate(graph_chunks, start=1):
        k = _key(chunk)
        scores[k]    = scores.get(k, 0.0) + graph_weight / (60 + rank)
        chunk_map[k] = chunk

    for rank, chunk in enumerate(vector_chunks, start=1):
        k = _key(chunk)
        scores[k] = scores.get(k, 0.0) + vector_weight / (60 + rank)
        if k not in chunk_map:
            chunk_map[k] = chunk

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    for k in sorted_keys:
        chunk_map[k]["blend_score"] = round(scores[k], 6)

    return [chunk_map[k] for k in sorted_keys]
