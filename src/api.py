"""
FastAPI application — D3 endpoints.
done by Baraa

Routes:
  POST /ingest        — trigger PDF ingestion pipeline (calls ingest.py)
  POST /search        — Mode A: hybrid BM25+dense search (HybridSearch)
  POST /search/graph  — Mode B: graph-guided retrieval, no CrossEncoder
  POST /answer        — Mode C: full pipeline, CrossEncoder + Groq
  POST /feedback      — record user feedback (stored in MongoDB)
  GET  /stats         — collection counts across all stores
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Docker Compose injects env vars directly into the container — no .env file needed here.
# For running api.py locally (outside Docker), export vars manually or use .env.local.

_searcher  = None
_graph_rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _searcher, _graph_rag
    from src.vector_store import HybridSearch
    from src.graphrag import GraphRAGExecutor
    _searcher  = HybridSearch()
    _graph_rag = GraphRAGExecutor()
    yield
    if _graph_rag:
        _graph_rag.close()

app = FastAPI(title="CSAI415 D3 — GraphRAG Stack", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_STATIC):
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

@app.get("/", include_in_schema=False)
def root():
    idx = os.path.join(_STATIC, "index.html")
    if os.path.isfile(idx):
        return FileResponse(idx)
    return {"status": "GraphRAG API running"}


# ── request / response models ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    page: int
    text: str
    score: float

class GraphSearchRequest(BaseModel):
    query:  str
    top_k:  int  = 5
    rerank: bool = False

class AnswerRequest(BaseModel):
    query:  str
    top_k:  int  = 5

class AnswerResponse(BaseModel):
    chunks: list[dict]
    answer: str

class FeedbackRequest(BaseModel):
    query: str
    chunk_id: str
    helpful: bool


# ── routes ─────────────────────────────────────────────────────────────────

@app.post("/ingest")
def ingest(data_dir: str = "data"):
    """Trigger ingestion of all PDFs in data_dir into MongoDB + Qdrant."""
    try:
        from src.ingest import run_ingest
        n_chunks = run_ingest(data_dir=data_dir)
        return {"status": "ok", "chunks_inserted": n_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    """Hybrid BM25 + dense search over ingested chunks."""
    try:
        results = _searcher.search(req.query, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/graph")
def search_graph(req: GraphSearchRequest):
    """Mode B/C retrieval: graph-guided subgraph selection + optional CrossEncoder rerank."""
    try:
        chunks = _graph_rag.search(req.query, top_k=req.top_k, rerank=req.rerank)
        return {"chunks": chunks}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    """Mode C full pipeline: graph retrieval + CrossEncoder rerank + Groq answer generation."""
    try:
        result = _graph_rag.answer(req.query, top_k=req.top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/rebuild-index")
def rebuild_index():
    """Force-rebuild the in-memory BM25+dense index (call after /ingest adds new docs)."""
    global _searcher
    try:
        from src.vector_store import HybridSearch
        _searcher = HybridSearch()
        return {"status": "rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Record a relevance signal — stored in MongoDB, consumed by online learner in D3."""
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGO_URI", "mongodb://admin:changeme@localhost:27017")
        client = MongoClient(uri)
        client.d2.feedback.insert_one(req.model_dump())
        return {"status": "recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """Return document/chunk counts from MongoDB and Qdrant collection info."""
    out = {}
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGO_URI", "mongodb://admin:changeme@localhost:27017")
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        out["mongo_chunks"]    = client.d2.chunks.count_documents({})
        out["mongo_docs"]      = client.d2.docs.count_documents({})
        out["mongo_feedback"]  = client.d2.feedback.count_documents({})
    except Exception as e:
        out["mongo_error"] = str(e)

    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        col  = os.getenv("QDRANT_COLLECTION", "d2_chunks")
        qc   = QdrantClient(host=host, port=port)
        info = qc.get_collection(col)
        out["qdrant_vectors"] = qc.count(col).count
    except Exception as e:
        out["qdrant_error"] = str(e)

    return out
