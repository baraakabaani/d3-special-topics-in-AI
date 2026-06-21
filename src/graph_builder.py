"""
graph_builder.py — D3.
done by Khalid

Reads enriched paper metadata from MongoDB (d2.docs), extracts entities,
and loads them into a Neo4j knowledge graph.

Graph schema
------------
Nodes
  (:Paper)   – one node per academic paper
  (:Author)  – one node per unique author name
  (:Topic)   – one node per normalized keyword extracted from title

Relationships (base layer — built in load_all)
  (:Paper)-[:WROTE]->(:Author)         renamed from AUTHORED_BY in D2
  (:Paper)-[:HAS_TOPIC]->(:Topic)

Derived relationships (built in build_derived_edges after all papers loaded)
  (:Author)-[:CO_AUTHORED_WITH]->(:Author)
      Direct Author-Author edge for any pair sharing at least one Paper.
      Direction is canonical (a1.name < a2.name) to avoid duplicate pairs.

  (:Paper)-[:RELATED_TO {shared_topics: int}]->(:Paper)
      Direct Paper-Paper edge for any pair sharing >= 2 Topic nodes.
      Direction is canonical (p1.doc_id < p2.doc_id).
      Property shared_topics records the exact overlap count.

Usage:
  python -m src.graph_builder
"""

import logging
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient

load_dotenv(".env.local", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── connection details ────────────────────────────────────────────────────────
MONGO_URI  = os.getenv("MONGO_URI",  "mongodb://admin:changeme@localhost:27017/?authSource=admin")
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "changeme123")

# ── stopwords for topic extraction ────────────────────────────────────────────
STOPWORDS = {
    "with", "using", "from", "that", "this", "based", "large", "small",
    "model", "models", "learning", "neural", "network", "networks", "deep",
    "towards", "efficient", "approach", "method", "methods", "paper", "study",
    "analysis", "framework", "system", "systems", "data", "dataset", "bench",
    "evaluation", "survey", "review", "novel", "new", "improved", "simple",
    "better", "beyond", "without", "multi", "joint", "end", "scale", "high",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_topics(title: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{5,}", title)
    return list({w.lower() for w in words if w.lower() not in STOPWORDS})[:6]


def extract_authors(authors_str: str) -> list[str]:
    if not authors_str or authors_str == "Unknown":
        return []
    parts = re.split(r"[;,]", authors_str)
    return [a.strip() for a in parts if a.strip()]


# ── Cypher — base schema ──────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper)  REQUIRE p.doc_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name   IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic)  REQUIRE t.name   IS UNIQUE",
]

MERGE_PAPER = """
MERGE (p:Paper {doc_id: $doc_id})
SET   p.title   = $title,
      p.year    = $year,
      p.venue   = $venue,
      p.authors = $authors
"""

# D3: renamed from AUTHORED_BY → WROTE
MERGE_AUTHOR_REL = """
MATCH (p:Paper {doc_id: $doc_id})
MERGE (a:Author {name: $name})
MERGE (p)-[:WROTE]->(a)
"""

MERGE_TOPIC_REL = """
MATCH (p:Paper {doc_id: $doc_id})
MERGE (t:Topic {name: $name})
MERGE (p)-[:HAS_TOPIC]->(t)
"""

# ── Cypher — derived edges (run AFTER all papers are loaded) ──────────────────

# Direct Author-Author co-authorship edge.
# WHERE a1.name < a2.name prevents both (a1→a2) and (a2→a1) from being created.
# MERGE is idempotent: safe to re-run.
CO_AUTHORED_WITH_CYPHER = """
MATCH (a1:Author)<-[:WROTE]-(p:Paper)-[:WROTE]->(a2:Author)
WHERE a1.name < a2.name
MERGE (a1)-[:CO_AUTHORED_WITH]->(a2)
"""

# Paper-Paper similarity edge based on shared Topic nodes.
# Threshold >=2 keeps meaningfully related pairs (works for both small demos and full corpus).
# stored property shared_topics lets downstream queries weight by overlap.
RELATED_TO_CYPHER = """
MATCH (p1:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)
WHERE p1.doc_id < p2.doc_id
WITH p1, p2, count(t) AS shared_topics
WHERE shared_topics >= 2
MERGE (p1)-[r:RELATED_TO]->(p2)
SET r.shared_topics = shared_topics
"""

# ── Cypher — 8 example queries (5 original + 3 new multi-hop D3 queries) ─────

EXAMPLE_QUERIES = [
    (
        "1. Papers per year",
        "MATCH (p:Paper) RETURN p.year AS year, count(*) AS papers ORDER BY year DESC",
    ),
    (
        "2. Most prolific authors (top 10)",
        """
        MATCH (p:Paper)-[:WROTE]->(a:Author)
        RETURN a.name AS author, count(p) AS papers
        ORDER BY papers DESC LIMIT 10
        """,
    ),
    (
        "3. Most common topics (top 10)",
        """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
        RETURN t.name AS topic, count(p) AS papers
        ORDER BY papers DESC LIMIT 10
        """,
    ),
    (
        "4. Papers sharing the topic 'agent' (top 5)",
        """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic {name: 'agent'})
        RETURN p.doc_id AS doc_id, p.title AS title
        LIMIT 5
        """,
    ),
    (
        "5. Co-author pairs via shared Paper (top 10)",
        """
        MATCH (a1:Author)<-[:WROTE]-(p:Paper)-[:WROTE]->(a2:Author)
        WHERE a1.name < a2.name
        RETURN a1.name AS author1, a2.name AS author2, count(p) AS shared_papers
        ORDER BY shared_papers DESC LIMIT 10
        """,
    ),
    (
        "6. [D3] Direct CO_AUTHORED_WITH neighbours of most prolific author",
        """
        MATCH (p:Paper)-[:WROTE]->(top:Author)
        WITH top, count(p) AS n ORDER BY n DESC LIMIT 1
        MATCH (top)-[:CO_AUTHORED_WITH]-(collaborator:Author)
        RETURN top.name AS author, collaborator.name AS collaborator
        ORDER BY collaborator.name
        """,
    ),
    (
        "7. [D3] Papers most closely RELATED_TO a seed paper (Text-Graph Synergy)",
        """
        MATCH (seed:Paper {doc_id: '2605.05643v1'})-[r:RELATED_TO]-(related:Paper)
        RETURN related.doc_id AS doc_id,
               related.title  AS title,
               r.shared_topics AS shared_topics
        ORDER BY r.shared_topics DESC
        LIMIT 5
        """,
    ),
    (
        "8. [D3] Papers covering BOTH 'recommendation' AND 'agent' topics",
        """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t1:Topic {name: 'recommendation'})
        MATCH (p)-[:HAS_TOPIC]->(t2:Topic {name: 'agent'})
        RETURN p.doc_id AS doc_id, p.title AS title
        """,
    ),
]


# ── GraphBuilder ──────────────────────────────────────────────────────────────

class GraphBuilder:

    def __init__(self):
        self.mongo  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000).d2
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        log.info("Connected to MongoDB (d2) and Neo4j")

    def setup_constraints(self):
        with self.driver.session() as s:
            for q in CONSTRAINTS:
                s.run(q)
        log.info("Constraints verified")

    def load_all(self) -> dict:
        """Load Paper, Author, and Topic nodes + WROTE and HAS_TOPIC edges."""
        self.setup_constraints()

        docs = list(self.mongo.docs.find({}))
        log.info(f"Loading {len(docs)} papers into Neo4j...")

        paper_count = author_count = topic_count = 0

        for doc in docs:
            doc_id  = doc["doc_id"]
            title   = doc.get("title",   "Unknown")
            authors = doc.get("authors", "Unknown")
            year    = doc.get("year",    0)
            venue   = doc.get("venue",   "arXiv")

            with self.driver.session() as s:
                s.run(MERGE_PAPER, doc_id=doc_id, title=title,
                      year=year, venue=venue, authors=authors)
            paper_count += 1

            for author in extract_authors(authors):
                with self.driver.session() as s:
                    s.run(MERGE_AUTHOR_REL, doc_id=doc_id, name=author)
                author_count += 1

            for topic in extract_topics(title):
                with self.driver.session() as s:
                    s.run(MERGE_TOPIC_REL, doc_id=doc_id, name=topic)
                topic_count += 1

        stats = {"papers": paper_count, "authors": author_count, "topics": topic_count}
        log.info(f"Base graph loaded: {stats}")
        return stats

    def build_derived_edges(self) -> dict:
        """
        Build CO_AUTHORED_WITH and RELATED_TO edges in a single cleanup pass.

        Must be called AFTER load_all() has committed all Paper/Author/Topic
        nodes and WROTE/HAS_TOPIC edges. Running before load_all completes
        would create CO_AUTHORED_WITH edges over a partial author set.
        """
        log.info("Building derived edges (CO_AUTHORED_WITH, RELATED_TO)...")

        with self.driver.session() as s:
            r1 = s.run(CO_AUTHORED_WITH_CYPHER)
            s1 = r1.consume()
            coauth_created = s1.counters.relationships_created

        with self.driver.session() as s:
            r2 = s.run(RELATED_TO_CYPHER)
            s2 = r2.consume()
            related_created = s2.counters.relationships_created

        stats = {
            "CO_AUTHORED_WITH_created": coauth_created,
            "RELATED_TO_created":       related_created,
        }
        log.info(f"Derived edges built: {stats}")
        return stats

    def run_example_queries(self):
        print("\n── Example Cypher Queries ──────────────────────────────────────")
        with self.driver.session() as s:
            for label, query in EXAMPLE_QUERIES:
                print(f"\n{label}")
                rows = s.run(query).data()
                if not rows:
                    print("  (no results)")
                for row in rows:
                    print(" ", row)
        print("────────────────────────────────────────────────────────────────\n")

    def summary(self):
        counts = {
            "Paper nodes":           "MATCH (p:Paper)  RETURN count(p) AS n",
            "Author nodes":          "MATCH (a:Author) RETURN count(a) AS n",
            "Topic nodes":           "MATCH (t:Topic)  RETURN count(t) AS n",
            "WROTE edges":           "MATCH ()-[:WROTE]->()            RETURN count(*) AS n",
            "HAS_TOPIC edges":       "MATCH ()-[:HAS_TOPIC]->()        RETURN count(*) AS n",
            "CO_AUTHORED_WITH":      "MATCH ()-[:CO_AUTHORED_WITH]->() RETURN count(*) AS n",
            "RELATED_TO edges":      "MATCH ()-[:RELATED_TO]->()       RETURN count(*) AS n",
        }
        print("\n── Graph Summary ───────────────────────────────")
        with self.driver.session() as s:
            for label, q in counts.items():
                n = s.run(q).single()["n"]
                print(f"  {label:<24} {n:>6}")
        print("────────────────────────────────────────────────\n")

    def prune_topics(self, max_papers: int = 20) -> dict:
        """Remove Topic nodes appearing in more than max_papers papers."""
        cypher = """
        MATCH (t:Topic)<-[:HAS_TOPIC]-(p:Paper)
        WITH  t, count(p) AS freq
        WHERE freq > $max_papers
        DETACH DELETE t
        """
        with self.driver.session() as session:
            result  = session.run(cypher, max_papers=max_papers)
            summary = result.consume()
            deleted = {
                "nodes":         summary.counters.nodes_deleted,
                "relationships": summary.counters.relationships_deleted,
            }
        log.info(f"Pruned {deleted['nodes']} over-represented topic(s), "
                 f"{deleted['relationships']} edges removed (threshold: >{max_papers} papers)")
        return deleted

    def close(self):
        self.driver.close()
        self.mongo.client.close()


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    builder = GraphBuilder()
    try:
        builder.load_all()
        builder.prune_topics()
        builder.build_derived_edges()   # must run after load_all
        builder.summary()
        builder.run_example_queries()
    finally:
        builder.close()
