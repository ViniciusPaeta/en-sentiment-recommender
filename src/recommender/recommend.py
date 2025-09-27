"""
CONTENT-BASED RECOMMENDER (heavily commented)

Given a query:
1) Transform to TF-IDF with the same vectorizer as the corpus
2) Compute cosine similarity against the TF-IDF matrix
3) Return top-K (index, score, text)
"""

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = "artifacts"
IDX_PATH = f"{ARTIFACTS_DIR}/tfidf_matrix.npz"
VECT_PATH = f"{ARTIFACTS_DIR}/tfidf_vectorizer.joblib"
CORPUS_PATH = f"{ARTIFACTS_DIR}/corpus.joblib"

_vect = _corpus = _X = None


def _load_all() -> None:
    """Lazy-load artifacts only once."""
    global _vect, _corpus, _X
    if _vect is None:
        _vect = joblib.load(VECT_PATH)
        _corpus = joblib.load(CORPUS_PATH)
        _X = sparse.load_npz(IDX_PATH)


def recommend(query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
    """
    Return top-K similar documents to a free-text query.
    """
    _load_all()
    qv = _vect.transform([query])
    sims = cosine_similarity(qv, _X).ravel()
    idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i]), _corpus[i]) for i in idx]
