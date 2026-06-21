"""
Data loading utilities.
- If corpus/queries/qrels JSON exist in data_dir  -> load them directly.
- If only PDFs exist in data_dir                  -> ingest PDFs, save JSON, load.
- If nothing real exists                          -> synthetic fallback (dev only).


"""

import json
import os
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# PDF ingestion (no separate module — avoids all import-path issues)
# ---------------------------------------------------------------------------

def _pdf_to_text(pdf_path: str, max_pages: int = 3) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = " ".join(
            (p.extract_text() or "") for p in reader.pages[:max_pages]
        )
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        print(f"  WARN {os.path.basename(pdf_path)}: {e}")
        return ""


def _guess_title(text: str, fallback: str) -> str:
    for chunk in text.split("."):
        chunk = chunk.strip()
        if 8 < len(chunk) < 120 and not chunk.lower().startswith("http"):
            return chunk
    return fallback


def _build_qrels(texts: list, n_rel: int = 4) -> list:
    """Top-n_rel most similar docs per document (TF-IDF cosine), excluding self."""
    vec = TfidfVectorizer(max_features=8000, stop_words="english")
    mat = vec.fit_transform(texts)
    sim = cosine_similarity(mat)
    qrels = []
    for i in range(len(texts)):
        row = sim[i].copy()
        row[i] = -1
        top = np.argsort(row)[::-1][:n_rel]
        qrels.append(sorted(top.tolist()))
    return qrels


def _ingest(pdf_dir: str, out_dir: str) -> None:
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    print(f"Ingesting {len(pdfs)} PDFs from {pdf_dir} ...")

    corpus, titles = [], []
    for i, fname in enumerate(pdfs):
        text = _pdf_to_text(os.path.join(pdf_dir, fname))
        if not text:
            continue
        title = _guess_title(text, fname.replace(".pdf", ""))
        titles.append(title)
        corpus.append(f"{title}. {text[:3000]}")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pdfs)} done")

    _TEMPLATES = [
        lambda t: t.lower(),
        lambda t: "survey of " + t.lower(),
        lambda t: t.lower() + " methods",
        lambda t: "evaluation of " + t.lower(),
        lambda t: t.lower() + " for retrieval",
    ]
    queries = [_TEMPLATES[i % len(_TEMPLATES)](titles[i]) for i in range(len(titles))]
    qrels   = _build_qrels(corpus)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "corpus.json"),  "w", encoding="utf-8") as f:
        json.dump(corpus,  f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "qrels.json"),   "w", encoding="utf-8") as f:
        json.dump(qrels,   f, indent=2)

    print(f"Saved {len(corpus)} docs / {len(queries)} queries -> {out_dir}/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(data_dir: str = "data"):
    """
    Returns (corpus: list[str], queries: list[str], qrels: list[list[int]]).
    Auto-ingests PDFs if JSON files are absent.
    """
    cp = os.path.join(data_dir, "corpus.json")
    qp = os.path.join(data_dir, "queries.json")
    rp = os.path.join(data_dir, "qrels.json")

    if all(os.path.exists(p) for p in [cp, qp, rp]):
        with open(cp, encoding="utf-8") as f: corpus  = json.load(f)
        with open(qp, encoding="utf-8") as f: queries = json.load(f)
        with open(rp, encoding="utf-8") as f: qrels   = json.load(f)
        print(f"Loaded: {len(corpus)} docs, {len(queries)} queries "
              f"(avg {sum(len(r) for r in qrels)/len(qrels):.1f} relevant/query)")
        return corpus, queries, qrels

    # Auto-ingest if PDFs are in data_dir
    if os.path.isdir(data_dir):
        pdfs = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
        if pdfs:
            _ingest(pdf_dir=data_dir, out_dir=data_dir)
            return load_data(data_dir)

    # Synthetic fallback (dev only — no real data present)
    print("WARNING: no real data found, using synthetic fallback")
    return _synthetic()


# ---------------------------------------------------------------------------
# Synthetic fallback (kept only for offline dev without PDFs)
# ---------------------------------------------------------------------------

def _synthetic(n_docs=300, n_queries=60, seed=42):
    rng = random.Random(seed)
    topics = {
        "ml":  {"name": "machine learning",          "core": ["gradient descent","overfitting","regularization","cross-validation","hyperparameter"],          "shared": ["model","training","prediction","dataset"]},
        "ir":  {"name": "information retrieval",     "core": ["BM25","inverted index","query expansion","term frequency","document ranking"],                   "shared": ["query","document","ranking","retrieval"]},
        "nlp": {"name": "natural language processing","core": ["tokenization","word embeddings","transformer","attention","language model"],                     "shared": ["text","token","sentence","model"]},
        "rec": {"name": "recommendation systems",    "core": ["collaborative filtering","matrix factorization","cold start","implicit feedback","embedding"],   "shared": ["user","item","rating","prediction"]},
        "se":  {"name": "search engines",            "core": ["web crawling","PageRank","anchor text","index compression","query processing"],                  "shared": ["query","document","ranking","index"]},
    }
    keys = list(topics.keys())
    corpus, doc_topic = [], []
    for i in range(n_docs):
        pk = keys[i % len(keys)]
        sk = rng.choice([k for k in keys if k != pk])
        pt, st = topics[pk], topics[sk]
        kws = rng.sample(pt["core"], 2)
        corpus.append(f"{pt['name']} paper. We study {kws[0]} and {kws[1]}. "
                       f"Uses {rng.choice(st['shared'])} from {st['name']}. Doc-{i}.")
        doc_topic.append(pk)
    t2d = {k: [i for i,t in enumerate(doc_topic) if t==k] for k in keys}
    queries, qrels = [], []
    for _ in range(n_queries):
        tk = rng.choice(keys)
        kw = rng.choice(topics[tk]["core"])
        queries.append(f"{kw} methods")
        qrels.append(sorted(rng.sample(t2d[tk], min(4, len(t2d[tk])))))
    return corpus, queries, qrels
