import os
import time
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from src.data.load_imdb import load_data


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "sentiment_tfidf_logreg.joblib"
REPORT_PATH = ARTIFACTS_DIR / "baseline_report.txt"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"


def _maybe_sample(X, y, n, name: str):
    """
    Optionally subsample (X, y) to n examples for faster prototyping.
    Uses a fixed RNG for reproducibility. No stratification (quick & simple).
    """
    if not n or n <= 0 or n >= len(X):
        return X, y
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=int(n), replace=False)
    # iloc preserves order of selected indices; reset_index cleans the index
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def main():
    """
    Train a baseline Sentiment Classifier: TF-IDF (1-2 grams) + Logistic Regression.
    - Loads IMDB (train/test) via our robust loader.
    - Fits the pipeline.
    - Prints metrics and saves artifacts (model + text report + JSON metrics).
    """

    t0 = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ===== 1) Load data =====
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    # Optional sampling via environment variables (useful to iterate fast)
    train_sample = int(os.getenv("BASELINE_TRAIN_SAMPLE", "0"))
    test_sample = int(os.getenv("BASELINE_TEST_SAMPLE", "0"))
    X_train, y_train = _maybe_sample(X_train, y_train, train_sample, "train")
    X_test, y_test = _maybe_sample(X_test, y_test, test_sample, "test")

    print(f"Train: {len(X_train)} docs | Test: {len(X_test)} docs")

    # ===== 2) Build pipeline =====
    # Notes:
    # - stop_words="english" removes common English words (less noise).
    # - sublinear_tf=True applies log-scaling to term frequencies (can help LR).
    # - dtype=np.float32 reduces memory footprint of TF-IDF matrix (default is float64).
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )

    # For binary classification with L2 penalty, 'lbfgs' is a solid default.
    # random_state is only used by some solvers (e.g., 'saga'), but keeping it is harmless.
    clf = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        random_state=42,
    )

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    # ===== 3) Train =====
    print("Fitting model...")
    pipe.fit(X_train, y_train)

    # ===== 4) Evaluate =====
    print("Evaluating...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print(report)
    print(f"Accuracy: {acc:.4f}")

    # ===== 5) Save artifacts =====
    joblib.dump(pipe, MODEL_PATH, compress=3)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + f"\nAccuracy: {acc:.4f}\n")
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)

    dt = time.time() - t0
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"📝 Report saved to: {REPORT_PATH}")
    print(f"⏱️ Total time: {dt:.1f}s")


if __name__ == "__main__":
    main()
