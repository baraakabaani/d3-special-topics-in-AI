"""
seed_demo.py — D3.
done by Yousef

Seeds all three stores (MongoDB, Qdrant, Neo4j) with 5 representative papers
for the live demo. Wipes existing data first — re-run safe.

Pipeline
--------
  1. Health checks  (exponential backoff, max 5 retries)
  2. clear_existing_data()
  3. run_ingest()              → MongoDB d2.docs + d2.chunks
  4. embed_and_index()         → Qdrant  d2_chunks
  5. GraphBuilder full pass    → Neo4j (load_all → prune_topics → build_derived_edges)
  6. verify_environment()      → counts every node/edge type

Usage (from project root):
  python scripts/seed_demo.py
"""

import os
import shutil
import sys
import tempfile
import time

# Add project root (parent of scripts/) to sys.path so `src.*` imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(".env.local", override=True)

DEMO_DOC_IDS = [
    "2604.28142v1",
    "2605.00318v1",
    "2605.05643v1",
    "2605.05538v1",
    "2605.00063v1",
]
DATA_DIR = "data"


# ── health checks ─────────────────────────────────────────────────────────────

def _wait_for_service(name: str, check_fn, max_retries: int = 5, base_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            check_fn()
            print(f"  {name:<10} ready")
            return
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"  {name:<10} not ready (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"             retrying in {wait:.0f}s...")
                time.sleep(wait)
    print(f"  {name:<10} FAILED after {max_retries} attempts — is Docker running?")
    sys.exit(1)


def health_checks():
    print("\n[1/6] Health checks...")

    from neo4j import GraphDatabase
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    mongo_uri   = os.getenv("MONGO_URI",   "mongodb://admin:changeme@localhost:27017/?authSource=admin")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    neo4j_uri   = os.getenv("NEO4J_URI",   "bolt://localhost:7687")
    neo4j_user  = os.getenv("NEO4J_USER",  "neo4j")
    neo4j_pass  = os.getenv("NEO4J_PASS",  "changeme123")

    def check_mongo():
        MongoClient(mongo_uri, serverSelectionTimeoutMS=2000).admin.command("ping")

    def check_qdrant():
        QdrantClient(host=qdrant_host, port=qdrant_port).get_collections()

    def check_neo4j():
        drv = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        with drv.session() as s:
            s.run("RETURN 1").single()
        drv.close()

    _wait_for_service("MongoDB",  check_mongo)
    _wait_for_service("Qdrant",   check_qdrant)
    _wait_for_service("Neo4j",    check_neo4j)


# ── clear existing data ───────────────────────────────────────────────────────

def clear_existing_data():
    print("\n[2/6] Clearing existing data...")

    from neo4j import GraphDatabase
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    mongo_uri   = os.getenv("MONGO_URI",   "mongodb://admin:changeme@localhost:27017/?authSource=admin")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    neo4j_uri   = os.getenv("NEO4J_URI",   "bolt://localhost:7687")
    neo4j_user  = os.getenv("NEO4J_USER",  "neo4j")
    neo4j_pass  = os.getenv("NEO4J_PASS",  "changeme123")
    collection  = os.getenv("QDRANT_COLLECTION", "d2_chunks")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.d2.docs.drop()
    client.d2.chunks.drop()
    client.close()
    print("  MongoDB   d2.docs + d2.chunks dropped")

    qc = QdrantClient(host=qdrant_host, port=qdrant_port)
    if qc.collection_exists(collection):
        qc.delete_collection(collection)
        print(f"  Qdrant    '{collection}' deleted")
    else:
        print(f"  Qdrant    '{collection}' did not exist — skipped")

    drv = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    drv.close()
    print("  Neo4j     all nodes + edges deleted")


# ── seed pipeline ─────────────────────────────────────────────────────────────

def seed_pipeline():
    print("\n[3/6] Ingesting demo papers into MongoDB...")

    # Copy only the 5 demo PDFs to a temp dir so run_ingest sees just those
    tmpdir = tempfile.mkdtemp(prefix="d3_demo_")
    try:
        for doc_id in DEMO_DOC_IDS:
            src = os.path.join(DATA_DIR, f"{doc_id}.pdf")
            dst = os.path.join(tmpdir, f"{doc_id}.pdf")
            if not os.path.exists(src):
                print(f"  WARN: {doc_id}.pdf not found in {DATA_DIR!r} — skipping")
                continue
            shutil.copy2(src, dst)
            print(f"  copied  {doc_id}.pdf")

        from src.ingest import run_ingest
        n_chunks = run_ingest(data_dir=tmpdir)
        print(f"  {n_chunks} chunks written to MongoDB")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n[4/6] Embedding + indexing into Qdrant...")
    from src.vector_store import embed_and_index
    n_vecs = embed_and_index(recreate=True)
    print(f"  {n_vecs} vectors indexed in Qdrant")

    print("\n[5/6] Building Neo4j knowledge graph...")
    builder = None
    try:
        from src.graph_builder import GraphBuilder
        builder = GraphBuilder()
        builder.load_all()
        builder.prune_topics(max_papers=20)
        builder.build_derived_edges()
        builder.summary()
    finally:
        if builder:
            builder.close()


# ── verification ──────────────────────────────────────────────────────────────

def verify_environment() -> bool:
    print("\n[6/6] Final verification...")

    from neo4j import GraphDatabase
    from pymongo import MongoClient
    from qdrant_client import QdrantClient

    mongo_uri   = os.getenv("MONGO_URI",   "mongodb://admin:changeme@localhost:27017/?authSource=admin")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    neo4j_uri   = os.getenv("NEO4J_URI",   "bolt://localhost:7687")
    neo4j_user  = os.getenv("NEO4J_USER",  "neo4j")
    neo4j_pass  = os.getenv("NEO4J_PASS",  "changeme123")
    collection  = os.getenv("QDRANT_COLLECTION", "d2_chunks")

    client   = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    n_docs   = client.d2.docs.count_documents({})
    n_chunks = client.d2.chunks.count_documents({})
    client.close()
    print(f"  MongoDB   docs={n_docs}  chunks={n_chunks}")

    qc     = QdrantClient(host=qdrant_host, port=qdrant_port)
    n_vecs = qc.count(collection).count
    print(f"  Qdrant    vectors={n_vecs}")

    drv = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    neo4j_counts = {}
    with drv.session() as s:
        for label, cypher in [
            ("Paper nodes",      "MATCH (p:Paper)  RETURN count(p) AS n"),
            ("Author nodes",     "MATCH (a:Author) RETURN count(a) AS n"),
            ("Topic nodes",      "MATCH (t:Topic)  RETURN count(t) AS n"),
            ("WROTE",            "MATCH ()-[:WROTE]->()            RETURN count(*) AS n"),
            ("HAS_TOPIC",        "MATCH ()-[:HAS_TOPIC]->()        RETURN count(*) AS n"),
            ("CO_AUTHORED_WITH", "MATCH ()-[:CO_AUTHORED_WITH]->() RETURN count(*) AS n"),
            ("RELATED_TO",       "MATCH ()-[:RELATED_TO]->()       RETURN count(*) AS n"),
        ]:
            neo4j_counts[label] = s.run(cypher).single()["n"]
    drv.close()

    for label, n in neo4j_counts.items():
        flag = "OK  " if n > 0 else "WARN"
        print(f"  [{flag}] Neo4j  {label:<22} = {n}")

    ok = (
        n_docs   >= len(DEMO_DOC_IDS)
        and n_chunks > 0
        and n_vecs   > 0
        and neo4j_counts.get("Paper nodes", 0) >= len(DEMO_DOC_IDS)
    )
    print("\n  Demo seed complete — stack is ready." if ok
          else "\n  WARN: some counts lower than expected — check logs above.")
    return ok


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    health_checks()
    clear_existing_data()
    seed_pipeline()
    ok = verify_environment()
    sys.exit(0 if ok else 1)
