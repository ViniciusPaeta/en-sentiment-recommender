import sys
import joblib
from pathlib import Path
from typing import List

MODEL_PATH = Path("artifacts/sentiment_tfidf_logreg.joblib")

def load_model():
    """
    Load the trained TF-IDF + Logistic Regression pipeline from disk.
    The pipeline includes both the vectorizer and the classifier.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.resolve()}. "
            f"Train it first: python -m src.models.train_baseline"
        )
    return joblib.load(MODEL_PATH)

def predict(texts: List[str]) -> List[int]:
    """
    Predict sentiment labels for a list of raw texts.
    Returns 0 (negative) or 1 (positive).
    """
    pipe = load_model()
    preds = pipe.predict(texts)
    # Ensure Python ints (not numpy types)
    return [int(x) for x in preds]

if __name__ == "__main__":
    # Usage:
    #   python -m src.models.predict "i loved it" "this was awful"
    if len(sys.argv) < 2:
        print("Usage: python -m src.models.predict <text1> [<text2> ...]")
        sys.exit(1)

    texts = sys.argv[1:]
    labels = predict(texts)
    for t, y in zip(texts, labels):
        print(f"[{y}] {t}")
