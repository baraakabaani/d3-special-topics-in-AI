"""
arXiv metadata enrichment — D2.
done by Yousef

Reads all docs from MongoDB, calls the arXiv API to get real titles,
authors, and abstracts, then updates the docs collection in-place.

Usage (from project root):
  python -m src.enrich
"""

import os
import re
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(".env.local", override=True)

MONGO_URI  = os.getenv("MONGO_URI", "mongodb://admin:changeme@localhost:27017/?authSource=admin")
ARXIV_API  = "http://export.arxiv.org/api/query"
BATCH_SIZE = 20
NS = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
}


def _strip_version(doc_id: str) -> str:
    """2605.00257v1 -> 2605.00257"""
    return re.sub(r"v\d+$", "", doc_id)


def _fetch_batch(arxiv_ids: list[str]) -> dict:
    """
    Call arXiv API for up to 20 IDs at once.
    Returns {clean_id: {title, authors, abstract}}.
    """
    resp = requests.get(
        ARXIV_API,
        params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)},
        timeout=20,
    )
    resp.raise_for_status()

    root    = ET.fromstring(resp.text)
    results = {}

    for entry in root.findall("atom:entry", NS):
        id_elem = entry.find("atom:id", NS)
        if id_elem is None:
            continue
        m = re.search(r"abs/(.+)", id_elem.text or "")
        if not m:
            continue
        clean_id = re.sub(r"v\d+$", "", m.group(1))

        title_elem   = entry.find("atom:title",   NS)
        summary_elem = entry.find("atom:summary", NS)
        title    = (title_elem.text   or "").strip().replace("\n", " ")
        abstract = (summary_elem.text or "").strip().replace("\n", " ")
        authors  = [
            a.find("atom:name", NS).text
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]

        results[clean_id] = {
            "title":    title,
            "authors":  ", ".join(authors),
            "abstract": abstract,
        }

    return results


def run_enrich() -> int:
    """
    Enrich all docs in MongoDB with real arXiv metadata.
    Returns the number of docs successfully updated.
    """
    db   = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000).d2
    docs = list(db.docs.find({}, {"doc_id": 1}))
    print(f"Enriching {len(docs)} docs from arXiv API...")

    id_map  = {_strip_version(d["doc_id"]): d["doc_id"] for d in docs}
    all_ids = list(id_map.keys())
    updated = 0

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i : i + BATCH_SIZE]

        for attempt in range(3):
            try:
                data = _fetch_batch(batch)
                break
            except Exception as e:
                print(f"  WARN  batch {i // BATCH_SIZE} attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"  SKIP  batch {i // BATCH_SIZE} after 3 failed attempts")
            continue

        for clean_id, meta in data.items():
            orig_id = id_map.get(clean_id)
            if not orig_id:
                continue
            db.docs.update_one(
                {"doc_id": orig_id},
                {"$set": {
                    "title":    meta["title"],
                    "authors":  meta["authors"],
                    "abstract": meta["abstract"],
                }},
            )
            updated += 1

        done = min(i + BATCH_SIZE, len(all_ids))
        print(f"  [{done:3d}/{len(all_ids)}]  +{len(data)} enriched  (total={updated})")
        time.sleep(0.4)   # stay under arXiv's 3 req/sec limit

    print(f"\nDone. {updated}/{len(docs)} docs enriched.")
    print(f"Sample:")
    for doc in db.docs.find({"abstract": {"$exists": True}}).limit(3):
        print(f"  {doc['doc_id']}  |  {doc['title'][:60]}")
        print(f"    authors: {doc['authors'][:60]}")
    return updated


if __name__ == "__main__":
    run_enrich()
