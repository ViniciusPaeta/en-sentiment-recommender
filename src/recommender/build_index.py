"""
TF-IDF INDEX BUILDER (heavily commented)

Creates:
- artifacts/tfidf_vectorizer.joblib : fitted TfidfVectorizer
- artifacts/corpus.joblib          : list[str] of training texts
- artifacts/tfidf_matrix.npz       : SciPy CSR sparse matrix
"""

import os

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.load_imdb import load_data

ARTIFACTS_DIR = "artifacts"
IDX_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.npz")
VECT_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
CORPUS_PATH = os.path.join(ARTIFACTS_DIR, "corpus.joblib")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train, _ = load_data("train")

    vect = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=2)
    X = vect.fit_transform(X_train)

    joblib.dump(vect, VECT_PATH)
    joblib.dump(list(X_train), CORPUS_PATH)
    sparse.save_npz(IDX_PATH, X)

    print("✅ TF-IDF index built and saved.")


if __name__ == "__main__":
    main()
