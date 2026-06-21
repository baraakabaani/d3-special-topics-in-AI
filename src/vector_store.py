"""
vector_store.py — D2.
done by Salim

Two responsibilities:
  1. embed_and_index() — reads all chunks from MongoDB (d2.chunks),
     embeds with bge-small-en-v1.5, upserts into Qdrant.
     Run once after ingest. Safe to re-run (idempotent).

  2. HybridSearch — BM25 + dense retrieval fused with Reciprocal Rank
     Fusion (RRF). Imported by api.py for the /search route.

Usage (from project root):
  python -m src.vector_store
"""

import os
from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

import numpy as np
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://admin:changeme@localhost:27017/?authSource=admin")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "d2_chunks")
MODEL_NAME  = "BAAI/bge-small-en-v1.5"
DIM         = 384
BATCH       = 32    # 128 causes 413 — Qdrant payload limit

# BGE models perform better with this prefix on passage text
BGE_PREFIX  = "Represent this sentence for searching relevant passages: "


def _get_clients():
    mongo  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000).d2
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return mongo, qdrant


def _get_chunks() -> list[dict]:
    mongo, _ = _get_clients()
    return list(mongo.chunks.find({}, {"_id": 0}))


def embed_and_index(recreate: bool = False) -> int:
    """
    Embed all MongoDB chunks and upsert into Qdrant.
    Uses sequential int IDs (not chunk_id strings) because Qdrant
    requires unsigned int64 or UUID — MD5 hex strings are neither.
    The chunk_id is stored in the payload for lookup.
    """
    mongo, qdrant = _get_clients()
    model = SentenceTransformer(MODEL_NAME)

    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION}'")

    if qdrant.collection_exists(COLLECTION) and not recreate:
        count = qdrant.count(COLLECTION).count
        if count > 0:
            print(f"Collection already has {count} vectors — skipping. Pass recreate=True to force.")
            return count

    chunks = list(mongo.chunks.find({}, {"_id": 0}))
    print(f"Embedding {len(chunks)} chunks into Qdrant...")

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts = [BGE_PREFIX + c["text"] for c in batch]
        vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            PointStruct(
                id=i + j,           # sequential int — Qdrant-safe
                vector=vecs[j].tolist(),
                payload={
                    "chunk_id":    c["chunk_id"],
                    "doc_id":      c["doc_id"],
                    "title":       c["title"],
                    "authors":     c["authors"],
                    "year":        c["year"],
                    "page_start":  c["page_start"],
                    "page_end":    c["page_end"],
                    "chunk_index": c["chunk_index"],
                    "text":        c["text"],
                },
            )
            for j, c in enumerate(batch)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points)
        print(f"  [{min(i + BATCH, len(chunks))}/{len(chunks)}] indexed")

    total = qdrant.count(COLLECTION).count
    print(f"Done. {total} vectors in Qdrant.")
    return total


class HybridSearch:
    """
    BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF).
    Imported by api.py for the /search route.
    """

    def __init__(self, rrf_k: int = 60, dense_pool: int = 100, bm25_pool: int = 100):
        self.rrf_k      = rrf_k
        self.dense_pool = dense_pool
        self.bm25_pool  = bm25_pool

        self._model   = SentenceTransformer(MODEL_NAME)
        self._chunks  = _get_chunks()
        self._id_map  = {c["chunk_id"]: c for c in self._chunks}
        tokenized     = [c["text"].lower().split() for c in self._chunks]
        self._bm25    = BM25Okapi(tokenized)
        self._ids     = [c["chunk_id"] for c in self._chunks]

        _, self._qdrant = _get_clients()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        dense_ids = self._dense_search(query, k=self.dense_pool)
        bm25_ids  = self._bm25_search(query,  k=self.bm25_pool)
        raw = self._rrf(dense_ids, bm25_ids)
        return self._aggregate_by_doc(raw)[:top_k]

    @staticmethod
    def _aggregate_by_doc(chunks: list[dict]) -> list[dict]:
        """Keep only the highest-scoring chunk per doc_id, then re-sort."""
        best: dict[str, dict] = {}
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in best or chunk["score"] > best[doc_id]["score"]:
                best[doc_id] = chunk
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    def _dense_search(self, query: str, k: int) -> list[str]:
        vec = self._model.encode(
            BGE_PREFIX + query, normalize_embeddings=True
        ).tolist()
        hits = self._qdrant.search(
            collection_name=COLLECTION, query_vector=vec, limit=k
        )
        return [h.payload["chunk_id"] for h in hits]

    def _bm25_search(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(query.lower().split())
        top    = np.argsort(scores)[::-1][:k]
        return [self._ids[i] for i in top]

    def _rrf(self, dense_ids: list[str], bm25_ids: list[str]) -> list[dict]:
        rrf_scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, cid in enumerate(bm25_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                **self._id_map[cid],
                "page":      self._id_map[cid]["page_start"],
                "score":     round(score, 6),
                "rrf_score": round(score, 6),
            }
            for cid, score in ranked
            if cid in self._id_map
        ]


if __name__ == "__main__":
    embed_and_index()
