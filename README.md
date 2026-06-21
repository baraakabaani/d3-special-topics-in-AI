# CSAI415 — Deliverable 3: GraphRAG Stack

**Team:** Baraa · Yousef · Khalid · Salim · Faisal

---

## Stack Overview

| Component | Technology | Purpose |
|---|---|---|
| Document store | MongoDB | Raw chunks + enriched metadata |
| Vector index | Qdrant | 384-dim BGE embeddings (2 640 vectors) |
| Knowledge graph | Neo4j | Papers, Authors, Topics with relationships |
| API | FastAPI | `/ingest`, `/search`, `/search/graph`, `/answer`, `/feedback`, `/stats` |
| Containerisation | Docker Compose | One-command stack bring-up |

---

## Architecture — Dataflow

```
  ┌─────────────┐
  │  PDF Files  │  (100 arXiv papers, RAG/IR domain)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     text + chunks      ┌──────────────────┐
  │  ingest.py  │ ─────────────────────► │  MongoDB (d2.*)  │
  │  pypdf      │                        │  chunks  2 640   │
  └──────┬──────┘                        │  docs      100   │
         │                               └──────────────────┘
         │  BGE-small embeddings
         ▼
  ┌──────────────────┐
  │  vector_store.py │ ── 384-dim vectors ──► Qdrant  2 640 pts
  └──────────────────┘
         │
         │  arXiv metadata (enrich.py)
         ▼
  ┌──────────────────┐
  │  graph_builder.py│ ──► Neo4j: 100 Papers · 510 Authors · 372 Topics
  └──────────────────┘

  ─────────────── Query Time ───────────────

  User Query
       │
       ├──► BM25 (rank-bm25, MongoDB chunks) ──┐
       │                                        ├──► RRF fusion ──► Top-k results
       └──► Dense (BGE + Qdrant ANN) ───────────┘    + doc aggregation
```

---

## How to Run

### 1. Prerequisites
- Docker Desktop running
- `.env.local` with credentials (see `.env.local.example`)
- PDFs in `data/`

### 2. Start the stack
```bash
docker compose up -d
```

### 3. Ingest PDFs
```bash
curl -X POST http://localhost:8000/ingest
```

### 4. Enrich metadata from arXiv
```bash
python -m src.enrich
```

### 5. Build Qdrant vectors
```bash
python -m src.vector_store
```

### 6. Build Neo4j graph
```bash
python -m src.graph_builder
```

### 7. Search
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "retrieval augmented generation survey", "top_k": 5}'
```

---

## Neo4j — 5 Example Cypher Queries

**1. Papers per year**
```cypher
MATCH (p:Paper)
RETURN p.year AS year, count(p) AS papers
ORDER BY year DESC
```
Result: `year: 2026 | papers: 100`

---

**2. Most prolific authors**
```cypher
MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
RETURN a.name AS author, count(p) AS papers
ORDER BY papers DESC LIMIT 10
```
Result (top 3): Wataru Uegami (2), Xi Wang (2), Tingyu Song (2)

---

**3. Most common topics**
```cypher
MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
RETURN t.name AS topic, count(p) AS papers
ORDER BY papers DESC LIMIT 10
```
Result (top 5): generation (16), augmented (14), aware (13), recommendation (13), agent (10)

---

**4. Papers sharing topic 'agent'**
```cypher
MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic {name: 'agent'})
RETURN p.doc_id AS doc_id, p.title AS title
LIMIT 5
```
Result: Superintelligent Retrieval Agent, AgenticRAG, A Case-Driven Multi-Agent Framework...

---

**5. Co-author pairs**
```cypher
MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
WHERE a1.name < a2.name
RETURN a1.name AS author1, a2.name AS author2, count(p) AS shared_papers
ORDER BY shared_papers DESC LIMIT 10
```
Result (top pair): Saghir Alfasly & Wataru Uegami (2 shared papers)

---

## Search — Top-k Examples with Citations

**Query: "retrieval augmented generation survey"**
1. *Seeking Information with RAG-Assistants: Does Model Size Matter?* — doc=2605.00964v1, p.13, score=0.0306
2. *Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning* — doc=2605.01399v1, p.10, score=0.0280
3. *AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases* — doc=2605.05538v1, p.8, score=0.0279

**Query: "dense passage retrieval transformers"**
1. *A Hybrid Retrieval and Reranking Framework for Evidence-Grounded RAG* — doc=2605.01664v1, p.2, score=0.0292
2. *Text-Graph Synergy: A Bidirectional Verification and Completion Framework* — doc=2605.05643v1, p.11, score=0.0291
3. *Efficient Multivector Retrieval with Token-Aware Clustering* — doc=2604.28142v1, p.6, score=0.0290

**Query: "LLM evaluation reasoning tasks"**
1. *OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers* — doc=2605.06235v1, p.5, score=0.0284
2. *LLM-Oriented Information Retrieval: A Denoising-First Perspective* — doc=2605.00505v1, p.1, score=0.0273
3. *A Survey of Reasoning-Intensive Retrieval: Progress and Challenges* — doc=2605.00063v1, p.2, score=0.0250

---

## Evaluation Metrics

Evaluated on 10 pseudo-relevance queries (query crafted from known paper title).

| Metric | Value |
|---|---|
| Recall@1 | **0.800** |
| Recall@3 | **0.900** |
| Recall@5 | **0.900** |
| Mean latency | **84 ms** |
| Std latency | 65 ms |

Reproduce:
```bash
python -m src.evaluate
```

---

## Team Contributions

| Member | File | Responsibility |
|---|---|---|
| Baraa | `Dockerfile`, `docker-compose.yml`, `src/api.py`, `notebooks/d2_demo.ipynb` | Infrastructure, API, demo notebook |
| Yousef | `src/ingest.py`, `src/enrich.py` | PDF ingestion, arXiv metadata enrichment |
| Khalid | `src/graph_builder.py`, `src/graph_queries.py` | Neo4j graph build + Cypher queries |
| Salim | `src/vector_store.py`, `src/search_demo.py` | Embeddings, hybrid search, citations demo |
| Faisal | `src/evaluate.py` | Recall@k evaluation, latency benchmarking |
