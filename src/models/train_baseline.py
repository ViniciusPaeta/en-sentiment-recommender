"""
BASELINE TRAINING SCRIPT (heavily commented)

What we train
-------------
A simple yet strong baseline for sentiment analysis:
- Vectorizer: TF-IDF with 1-2 grams, up to 50k features, min_df=2
- Classifier: Logistic Regression (good for sparse, high-dimensional text)

Artifacts
---------
- artifacts/sentiment_tfidf_logreg.joblib : the fitted Pipeline
- Console output: classification report on the test set

How to run
----------
    python -m src.models.train_baseline
"""

import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.data.load_imdb import load_data

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_tfidf_logreg.joblib")


def main() -> None:
    """
    Orchestrates data loading, training, evaluation and persistence.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=200, n_jobs=-1)),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("=== Classification Report (Test Set) ===")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
