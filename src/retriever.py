"""
HybridRetriever: BM25 + TruncatedSVD dense retrieval with score blending.
done by Faisal
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class HybridRetriever:
    def __init__(self, k=10, metric="cosine", svd_dim=128,
                 normalization="l2", hybrid_weight=0.5):
        self.k            = k
        self.metric       = metric
        self.svd_dim      = svd_dim
        self.normalization = normalization
        self.hybrid_weight = hybrid_weight   # alpha: 0=BM25 only, 1=dense only
        self._bm25     = None
        self._doc_vecs = None
        self._tfidf    = TfidfVectorizer()
        self._svd      = TruncatedSVD(n_components=svd_dim, random_state=42)

    def fit(self, corpus: list) -> None:
        tokenized = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

        tfidf_mat = self._tfidf.fit_transform(corpus)
        # clamp svd_dim to vocabulary size
        max_dim = tfidf_mat.shape[1] - 1
        if self._svd.n_components > max_dim:
            self._svd = TruncatedSVD(n_components=max_dim, random_state=42)
        vecs = self._svd.fit_transform(tfidf_mat)
        self._doc_vecs = self._norm(vecs)

    def retrieve(self, query: str, k: int = None) -> list:
        k = k or self.k
        bm25_raw   = np.array(self._bm25.get_scores(query.lower().split()))
        bm25_score = self._minmax(bm25_raw)

        q_tfidf    = self._tfidf.transform([query])
        q_vec      = self._norm(self._svd.transform(q_tfidf))[0]
        dense_score = self._dense(q_vec, self._doc_vecs)

        combined = self.hybrid_weight * dense_score + (1 - self.hybrid_weight) * bm25_score
        return np.argsort(combined)[::-1][:k].tolist()

    def _minmax(self, s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)

    def _norm(self, v):
        if self.normalization == "l2":     return normalize(v, norm="l2")
        if self.normalization == "minmax":
            lo, hi = v.min(0), v.max(0)
            rng = hi - lo; rng[rng == 0] = 1
            return (v - lo) / rng
        return v

    def _dense(self, q, D):
        if self.metric == "cosine":      return (D @ q + 1) / 2
        if self.metric == "dot_product": return D @ q
        if self.metric == "euclidean":   return 1 / (1 + np.linalg.norm(D - q, axis=1))
        raise ValueError(self.metric)
