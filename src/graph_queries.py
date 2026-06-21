"""
graph_queries.py — D3.
done by Khalid

Two responsibilities:
  1. select_subgraph() — called by GraphRAGExecutor to pick a ranked paper
     subgraph for a user query (two-hop: direct topic match + RELATED_TO expansion)
  2. run_queries() — prints the 8 demonstration Cypher queries for the report

Usage:
  python -m src.graph_queries
"""

import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(".env.local", override=True)

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "changeme123")

# Stopwords for keyword extraction (same set as graph_builder.py)
_STOP = {
    "with", "using", "from", "that", "this", "based", "large", "small",
    "model", "models", "learning", "neural", "network", "networks", "deep",
    "towards", "efficient", "approach", "method", "methods", "paper", "study",
    "analysis", "framework", "system", "systems", "data", "dataset", "bench",
    "evaluation", "survey", "review", "novel", "new", "improved", "simple",
    "better", "beyond", "without", "multi", "joint", "end", "scale", "high",
    "what", "does", "which", "papers", "corpus", "have", "their", "about",
}


def _extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a natural-language query for Cypher matching.

    Lowercases, keeps tokens >= 4 chars, removes stopwords.
    Returns at most 8 tokens so the Cypher IN list stays selective.
    """
    tokens = re.findall(r"[a-z]{4,}", query.lower())
    return [t for t in tokens if t not in _STOP][:8]


# ── Two-hop subgraph selection ────────────────────────────────────────────────

# Hop 1: papers whose Topic nodes match extracted keywords directly.
# Scored by number of matching topic nodes so a paper matching 4 of 5
# query keywords ranks higher than one matching only 1.
_HOP1_CYPHER = """
MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
WHERE t.name IN $keywords
WITH p, count(t) AS topic_matches
RETURN p.doc_id AS doc_id,
       p.title   AS title,
       topic_matches AS graph_score,
       1 AS hop
ORDER BY topic_matches DESC
LIMIT $limit
"""

# Hop 2: papers reachable from hop-1 seeds via RELATED_TO.
# Only returns papers NOT already in the hop-1 set.
# Scored by how many seed papers they are related to.
_HOP2_CYPHER = """
MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
WHERE t.name IN $keywords
WITH COLLECT(DISTINCT p.doc_id) AS seed_ids
MATCH (seed:Paper)-[:RELATED_TO]-(expanded:Paper)
WHERE seed.doc_id IN seed_ids
  AND NOT expanded.doc_id IN seed_ids
WITH expanded, count(seed) AS connection_count
RETURN expanded.doc_id AS doc_id,
       expanded.title   AS title,
       connection_count AS graph_score,
       2 AS hop
ORDER BY connection_count DESC
LIMIT $limit
"""


def select_subgraph(driver, query: str, limit: int = 15) -> list[dict]:
    """
    Two-hop subgraph selection for the GraphRAG executor.

    Hop 1: direct topic keyword matches, scored by matched topic count.
    Hop 2: RELATED_TO expansion from hop-1 papers, scored by connection count.
    Hop-1 results always rank above hop-2 results of equal score.

    Returns a list of dicts: [{doc_id, title, graph_score, hop}]
    ordered hop-1 first, then hop-2, each sub-sorted by score descending.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return []

    results: list[dict] = []
    seen_ids: set[str] = set()

    with driver.session() as s:
        for row in s.run(_HOP1_CYPHER, keywords=keywords, limit=limit):
            d = dict(row)
            results.append(d)
            seen_ids.add(d["doc_id"])

    with driver.session() as s:
        for row in s.run(_HOP2_CYPHER, keywords=keywords, limit=max(limit // 2, 5)):
            d = dict(row)
            if d["doc_id"] not in seen_ids:
                results.append(d)
                seen_ids.add(d["doc_id"])

    return results


# ── 8 demonstration queries ───────────────────────────────────────────────────

_DEMO_QUERIES = [
    (
        "1. Papers per year",
        """
        MATCH (p:Paper)
        RETURN p.year AS year, count(p) AS papers
        ORDER BY year DESC
        """,
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
        "6. [D3 multi-hop] CO_AUTHORED_WITH neighbours of most prolific author",
        """
        MATCH (p:Paper)-[:WROTE]->(top:Author)
        WITH top, count(p) AS n ORDER BY n DESC LIMIT 1
        MATCH (top)-[:CO_AUTHORED_WITH]-(collaborator:Author)
        MATCH (collaborator)<-[:WROTE]-(collab_paper:Paper)
        RETURN top.name      AS most_prolific_author,
               collaborator.name AS collaborator,
               collab_paper.title AS collaborator_paper
        ORDER BY collaborator.name
        LIMIT 10
        """,
    ),
    (
        "7. [D3 multi-hop] Papers most closely RELATED_TO a seed paper",
        """
        MATCH (seed:Paper {doc_id: '2605.05250v1'})-[r:RELATED_TO]-(related:Paper)
        RETURN related.doc_id   AS doc_id,
               related.title    AS title,
               r.shared_topics  AS shared_topics
        ORDER BY r.shared_topics DESC
        LIMIT 5
        """,
    ),
    (
        "8. [D3 multi-hop] Papers covering BOTH recommendation AND agent topics",
        # Synonym list handles stemming gap: 'recommender'/'recommendation' and 'agent'/'agentic'
        # are the same concept but extract as different tokens from titles.
        # TODO: fix properly in extract_topics() with stemming after D3 submission.
        """
        MATCH (p:Paper)-[:HAS_TOPIC]->(t1:Topic)
        WHERE t1.name IN ['recommendation', 'recommender']
        MATCH (p)-[:HAS_TOPIC]->(t2:Topic)
        WHERE t2.name IN ['agent', 'agentic']
        RETURN DISTINCT p.doc_id AS doc_id, p.title AS title
        """,
    ),
]


def run_queries():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        for title, cypher in _DEMO_QUERIES:
            print(f"\n{'='*60}")
            print(f"Query {title}")
            print('='*60)
            rows = session.run(cypher).data()
            if not rows:
                print("  (no results)")
            for row in rows:
                print("  " + "  |  ".join(f"{k}: {v}" for k, v in row.items()))

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    run_queries()
