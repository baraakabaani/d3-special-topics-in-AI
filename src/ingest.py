"""
PDF ingestion pipeline — D2.
done by Yousef

Pipeline:
  PDF -> full text extraction (all pages) -> overlapping chunks
       -> metadata extraction (title, authors, year, venue)
       -> MongoDB  d2.docs    (one doc per PDF)
       -> MongoDB  d2.chunks  (one doc per chunk, with page + provenance)

Usage (from project root):
  python -m src.ingest                        # ingests data/ folder
  python -m src.ingest --data-dir data/       # explicit path
"""

import argparse
import hashlib
import os
import re
import sys

from dotenv import load_dotenv

# host scripts load .env.local; override=True so stale shell vars don't shadow
load_dotenv(".env.local", override=True)

from pymongo import MongoClient, UpdateOne
from pypdf import PdfReader


# ── config from environment ─────────────────────────────────────────────────

MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://admin:changeme@localhost:27017")
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE",  512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


# ── text extraction ─────────────────────────────────────────────────────────

def _extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns a list of {page: int, text: str} for pages with extractable text.

    Blank/figure-only pages are skipped individually — they contain no text
    to chunk anyway and skipping them does not create retrieval gaps because
    the surrounding pages still provide full context.

    The entire PDF is only dropped (returns []) when ZERO pages yield text,
    which means it is a fully-scanned image PDF with no digital text layer.
    Dropping one blank page != dropping the document.
    """
    pages       = []
    total_pages = 0
    blank_pages = 0
    try:
        reader      = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            raw = page.extract_text()
            if not raw or not raw.strip():
                blank_pages += 1
                continue
            text = re.sub(r"\s+", " ", raw).strip()
            # strip lone surrogates that BSON/MongoDB cannot encode
            text = text.encode("utf-8", errors="ignore").decode("utf-8")
            if len(text) > 50:
                pages.append({"page": i + 1, "text": text})
            else:
                blank_pages += 1
    except Exception as e:
        print(f"  WARN  {os.path.basename(pdf_path)}: {e}")
        return []

    if blank_pages > 0 and pages:
        fname = os.path.basename(pdf_path)
        print(f"  INFO  {fname}: {blank_pages}/{total_pages} blank pages skipped, "
              f"{len(pages)} pages kept")

    return pages   # empty list = fully scanned, caller will skip the PDF


def _extract_metadata(pdf_path: str, pages: list[dict]) -> dict:
    """
    Best-effort metadata extraction.
    Priority: PDF metadata fields -> first-page heuristics -> filename fallback.
    """
    fname   = os.path.basename(pdf_path)
    doc_id  = fname.replace(".pdf", "")

    title, authors, year, venue = None, None, None, "arXiv"

    # ── PDF built-in metadata ──
    try:
        reader = PdfReader(pdf_path)
        meta   = reader.metadata or {}
        title  = (meta.get("/Title")  or "").strip() or None
        authors = (meta.get("/Author") or "").strip() or None
    except Exception:
        pass

    # ── year from arXiv ID format  YYMM.NNNNN ──
    m = re.match(r"(\d{2})(\d{2})\.\d+", doc_id)
    if m:
        year = int("20" + m.group(1))

    # ── title fallback: first meaningful sentence of page 1 ──
    if not title and pages:
        for sentence in pages[0]["text"].split("."):
            sentence = sentence.strip()
            if 10 < len(sentence) < 150 and not sentence.lower().startswith("http"):
                title = sentence
                break

    # ── final fallbacks ──
    if not title:
        title = doc_id
    if not authors:
        authors = "Unknown"
    if not year:
        year = 0

    return {
        "doc_id":  doc_id,
        "title":   title,
        "authors": authors,
        "year":    year,
        "venue":   venue,
        "source":  fname,
    }


# ── chunking ────────────────────────────────────────────────────────────────

def _build_word_page_map(pages: list[dict]) -> list[tuple[str, int]]:
    """
    Flatten all pages into a single list of (word, page_number) tuples.
    Chunking across this list gives correct cross-page overlap AND lets us
    record exact page_start/page_end for every chunk — including gaps where
    blank/figure pages were skipped (e.g. words jump from page 6 to page 8).
    """
    word_page = []
    for page_info in pages:
        for word in page_info["text"].split():
            word_page.append((word, page_info["page"]))
    return word_page


def _chunk_document(pages: list[dict], meta: dict,
                    size: int, overlap: int) -> list[dict]:
    """
    Produce chunk dicts ready for MongoDB insertion.

    Chunks are created across the full document (not page-by-page) so overlap
    works correctly at page boundaries. Each chunk records page_start and
    page_end from the actual page numbers of its first and last word —
    correctly handling gaps where blank pages were skipped.

    chunk_id is a deterministic MD5 hash so re-running ingest is idempotent.
    """
    word_page = _build_word_page_map(pages)
    if not word_page:
        return []

    chunks    = []
    start     = 0

    while start < len(word_page):
        end    = min(start + size, len(word_page))
        slice_ = word_page[start:end]

        text       = " ".join(w for w, _ in slice_)
        page_start = slice_[0][1]
        page_end   = slice_[-1][1]
        chunk_idx  = len(chunks)

        raw_id   = f"{meta['doc_id']}_{page_start}_{chunk_idx}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()

        chunks.append({
            "chunk_id":    chunk_id,
            "doc_id":      meta["doc_id"],
            "title":       meta["title"],
            "authors":     meta["authors"],
            "year":        meta["year"],
            "venue":       meta["venue"],
            "page_start":  page_start,
            "page_end":    page_end,
            "chunk_index": chunk_idx,
            "text":        text,
        })

        if end == len(word_page):
            break
        start += size - overlap

    return chunks


# ── MongoDB storage ──────────────────────────────────────────────────────────

def _get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client.d2
    # indexes — safe to call repeatedly (no-op if already exist)
    db.chunks.create_index("chunk_id", unique=True)
    db.chunks.create_index("doc_id")
    db.docs.create_index("doc_id", unique=True)
    return db


def _upsert_doc(db, meta: dict) -> None:
    db.docs.update_one(
        {"doc_id": meta["doc_id"]},
        {"$set": meta},
        upsert=True,
    )


def _upsert_chunks(db, chunks: list[dict]) -> int:
    if not chunks:
        return 0
    ops = [
        UpdateOne({"chunk_id": c["chunk_id"]}, {"$set": c}, upsert=True)
        for c in chunks
    ]
    result = db.chunks.bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


# ── public entry point ───────────────────────────────────────────────────────

def run_ingest(data_dir: str = "data") -> int:
    """
    Ingest all PDFs in data_dir into MongoDB.
    Returns total number of chunks inserted/updated.
    Called by api.py /ingest route and by the __main__ block below.
    """
    pdf_files = sorted(
        f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
        return 0

    print(f"Found {len(pdf_files)} PDFs in '{data_dir}'")
    print(f"Chunk size={CHUNK_SIZE} words, overlap={CHUNK_OVERLAP} words")

    db          = _get_db()
    total_chunks = 0
    skipped     = 0

    for i, fname in enumerate(pdf_files, 1):
        path  = os.path.join(data_dir, fname)
        pages = _extract_pages(path)

        if not pages:
            skipped += 1
            continue

        meta   = _extract_metadata(path, pages)
        chunks = _chunk_document(pages, meta, CHUNK_SIZE, CHUNK_OVERLAP)

        _upsert_doc(db, meta)
        n = _upsert_chunks(db, chunks)
        total_chunks += n

        if i % 10 == 0 or i == len(pdf_files):
            print(f"  [{i:3d}/{len(pdf_files)}] {fname[:50]:50s}  "
                  f"+{n} chunks  (total={total_chunks})")

    print(f"\nDone. {len(pdf_files) - skipped} PDFs ingested, "
          f"{skipped} skipped (no extractable text).")
    print(f"Total chunks in MongoDB: {db.chunks.count_documents({})}")
    return total_chunks


# ── verification helper ──────────────────────────────────────────────────────

def verify_chunks(sample_n: int = 5) -> None:
    """
    Sanity-check the ingested chunks:
    - No chunk shorter than 30 words
    - Every chunk has a non-null doc_id
    - Print sample_n example chunks
    """
    db = _get_db()
    total = db.chunks.count_documents({})
    short = db.chunks.count_documents(
        {"$expr": {"$lt": [{"$size": {"$split": ["$text", " "]}}, 30]}}
    )
    missing_doc_id = db.chunks.count_documents(
        {"$or": [{"doc_id": None}, {"doc_id": ""}]}
    )

    print(f"\nVerification report:")
    print(f"  Total chunks      : {total}")
    print(f"  Chunks < 30 words : {short}  {'OK' if short == 0 else 'WARNING'}")
    print(f"  Missing doc_id    : {missing_doc_id}  {'OK' if missing_doc_id == 0 else 'WARNING'}")
    # verify page ranges: page_end should always be >= page_start
    bad_ranges = db.chunks.count_documents(
        {"$expr": {"$lt": ["$page_end", "$page_start"]}}
    )
    print(f"  Bad page ranges   : {bad_ranges}  {'OK' if bad_ranges == 0 else 'WARNING'}")

    print(f"\nSample chunks:")
    for c in db.chunks.find().limit(sample_n):
        word_count  = len(c["text"].split())
        page_range  = (f"p{c['page_start']}" if c["page_start"] == c["page_end"]
                       else f"p{c['page_start']}-{c['page_end']}")
        print(f"  chunk_id={c['chunk_id'][:12]}  doc={c['doc_id']}  "
              f"{page_range:10s}  words={word_count}  "
              f"title={c['title'][:40]}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into MongoDB")
    parser.add_argument("--data-dir", default="data", help="Folder containing PDFs")
    parser.add_argument("--verify",   action="store_true",
                        help="Run verification checks after ingest")
    args = parser.parse_args()

    n = run_ingest(data_dir=args.data_dir)
    if args.verify or n > 0:
        verify_chunks()
