import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.data.load_imdb import load_data


# Where we store the recommender artifacts
RECO_DIR = Path("artifacts/recommender")
VECT_PATH = RECO_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = RECO_DIR / "tfidf_matrix.npz"
CORPUS_CSV = RECO_DIR / "corpus.csv"
META_JSON = RECO_DIR / "meta.json"


def _maybe_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Subsample the dataframe to n rows for quicker iterations.
    Uses a fixed RNG for reproducibility.
    """
    if not n or n <= 0 or n >= len(df):
        return df
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def main():
    """
    Build a TF-IDF index for content-based recommendations (cosine similarity).
    Steps:
      1) Load IMDB train/test via our robust loader and concatenate into one corpus.
      2) Fit a TF-IDF vectorizer (1–2-grams) with English stop-words and sublinear TF.
      3) Transform the corpus and L2-normalize the sparse matrix.
      4) Save vectorizer, matrix, and metadata to artifacts/recommender/.
    Environment knobs:
      - RECO_SAMPLE: optional integer to subsample the corpus for faster builds.
    """
    RECO_DIR.mkdir(parents=True, exist_ok=True)

    # ===== 1) Load corpus (train + test) =====
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    df_train = pd.DataFrame({"text": X_train, "label": y_train, "split": "train"})
    df_test = pd.DataFrame({"text": X_test, "label": y_test, "split": "test"})
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Optional sampling for quick iterations
    sample_n = int(os.getenv("RECO_SAMPLE", "0"))
    df = _maybe_sample(df, sample_n)

    print(f"Corpus size: {len(df)} documents")

    # ===== 2) Fit TF-IDF =====
    # Notes:
    # - ngram_range=(1, 2): unigrams + bigrams often work well for movies.
    # - min_df=2: drop very rare tokens.
    # - sublinear_tf: log-scale TF to reduce impact of very frequent terms.
    # - dtype=np.float32: reduce memory footprint.
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )

    print("Fitting TF-IDF vectorizer...")
    X_tfidf = vectorizer.fit_transform(df["text"].tolist())

    # ===== 3) L2-normalize for cosine similarity via dot product =====
    # With L2-normalization, cosine_sim(A, B) == dot(A, B).
    print("Normalizing TF-IDF matrix (L2)...")
    X_tfidf = normalize(X_tfidf, norm="l2", copy=False)

    # ===== 4) Persist artifacts =====
    print(f"Saving artifacts to: {RECO_DIR}/")
    joblib.dump(vectorizer, VECT_PATH, compress=3)
    sparse.save_npz(MATRIX_PATH, X_tfidf)

    # Save a lightweight corpus CSV (id, split, label, text)
    # id: row index aligned with TF-IDF rows for later lookup
    df_out = df.reset_index().rename(columns={"index": "id"})
    df_out.to_csv(CORPUS_CSV, index=False)

    meta = {
        "n_docs": int(X_tfidf.shape[0]),
        "n_features": int(X_tfidf.shape[1]),
        "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
        "ngram_range": [1, 2],
        "min_df": 2,
        "stop_words": "english",
        "sublinear_tf": True,
        "dtype": "float32",
        "normalized": "l2",
        "files": {
            "vectorizer": str(VECT_PATH),
            "matrix": str(MATRIX_PATH),
            "corpus_csv": str(CORPUS_CSV),
        },
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Recommender index built successfully.")
    print(f"- Vectorizer: {VECT_PATH}")
    print(f"- Matrix:     {MATRIX_PATH}  (shape={X_tfidf.shape[0]} x {X_tfidf.shape[1]})")
    print(f"- Corpus CSV: {CORPUS_CSV}")
    print(f"- Meta JSON:  {META_JSON}")


if __name__ == "__main__":
    main()
