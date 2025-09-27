"""
INFERENCE WRAPPER (heavily commented)

Purpose
-------
Offer a tiny API for inference that:
- Lazily loads the persisted sklearn Pipeline from disk (first call only)
- Caches it in-process for subsequent predictions
- Exposes a single `predict(texts)` function returning integer labels
"""

import os

import joblib

# Default path for the saved model pipeline
MODEL_PATH = os.path.join("artifacts", "sentiment_tfidf_logreg.joblib")

# Module-level cache so we don't hit disk more than once
_model_cache = None


def _get_model():
    """
    Returns the cached sklearn Pipeline, loading from disk if necessary.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def predict(texts: str | list[str]) -> list[int]:
    """
    Predict sentiment for a single string or a list of strings.

    Parameters
    ----------
    texts : str | list[str]
        Input review(s).

    Returns
    -------
    list[int]
        A list of integer labels (0=negative, 1=positive), one per input.
    """
    model = _get_model()
    batch: list[str] = texts if isinstance(texts, list) else [texts]
    return model.predict(batch).tolist()
