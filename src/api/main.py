from typing import Optional, List, Dict, Any
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator
from scipy import sparse
from sklearn.preprocessing import normalize

# Reuse our recommender utilities
from src.recommender.query import (
    load_index as load_reco_index,
    _vector_from_text,
    _vector_from_id,
    _apply_filters,
    recommend_from_vector,
)

APP_TITLE = "EN Sentiment + Recommender"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "sentiment_tfidf_logreg.joblib"

app = FastAPI(title=APP_TITLE)

# Globals loaded at startup
PIPE = None  # TF-IDF + LogisticRegression pipeline
VECT = None  # TfidfVectorizer
X_MAT: Optional[sparse.csr_matrix] = None  # L2-normalized TF-IDF matrix
CORPUS: Optional[pd.DataFrame] = None      # DataFrame with id/split/label/text


# ====== Schemas ======

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to classify")

class PredictOut(BaseModel):
    label: int
    proba: float

class RecommendIn(BaseModel):
    # Either text or id must be provided
    text: Optional[str] = Field(None, description="Query text to find similar items")
    id: Optional[int] = Field(None, ge=0, description="Existing doc_id to find neighbors of")
    k: int = Field(5, gt=0, le=100, description="Top-K results (1..100)")
    split: Optional[str] = Field(None, description="Optional filter: split (e.g., train/test)")
    label: Optional[int] = Field(None, ge=0, le=1, description="Optional filter: sentiment label 0/1")
    exclude_id: Optional[int] = Field(None, ge=0, description="Optional doc_id to be excluded")

    @root_validator
    def check_query_source(cls, values):
        text, _id = values.get("text"), values.get("id")
        if (text is None and _id is None) or (text and _id is not None):
            raise ValueError("Provide exactly one of: 'text' or 'id'.")
        return values

class RecommendItem(BaseModel):
    id: int
    score: float
    split: Optional[str] = None
    label: Optional[int] = None
    text: Optional[str] = None  # truncated preview

class RecommendOut(BaseModel):
    results: List[RecommendItem]


# ====== Startup: load model and recommender index ======

@app.on_event("startup")
def _load_artifacts() -> None:
    global PIPE, VECT, X_MAT, CORPUS

    # Load classifier pipeline for /predict
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH.resolve()}. "
            f"Train it first: python -m src.models.train_baseline"
        )
    PIPE = joblib.load(MODEL_PATH)

    # Load recommender index (vectorizer + normalized matrix + corpus)
    VECT, X_MAT, CORPUS = load_reco_index()
    if X_MAT is None or CORPUS is None:
        raise RuntimeError("Failed to load recommender index.")


# ====== Routes ======

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": PIPE is not None,
        "recommender_docs": int(X_MAT.shape[0]) if X_MAT is not None else 0,
        "features": int(X_MAT.shape[1]) if X_MAT is not None else 0,
    }

@app.post("/predict", response_model=PredictOut)
def predict_endpoint(item: PredictIn) -> PredictOut:
    """
    Predict sentiment label (0/1) and probability for the positive class.
    """
    if PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # proba for class 1 if available; fallback to decision function
        if hasattr(PIPE, "predict_proba"):
            proba = float(PIPE.predict_proba([item.text])[0, 1])
        elif hasattr(PIPE, "decision_function"):
            # Map decision scores to (0..1) via logistic; acceptable fallback
            score = float(PIPE.decision_function([item.text])[0])
            proba = float(1 / (1 + np.exp(-score)))
        else:
            # Last resort: hard label, proba as 0/1
            label = int(PIPE.predict([item.text])[0])
            return PredictOut(label=label, proba=float(label))
        label = int(PIPE.predict([item.text])[0])
        return PredictOut(label=label, proba=proba)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {type(e).__name__}: {e}")

@app.post("/recommend", response_model=RecommendOut)
def recommend_endpoint(q: RecommendIn) -> RecommendOut:
    """
    Return top-K similar items based on cosine similarity (TF-IDF, L2-normalized).
    Accepts either 'text' (raw query) or 'id' (existing document).
    Optional filters: split/label. Optional exclude_id.
    """
    if VECT is None or X_MAT is None or CORPUS is None:
        raise HTTPException(status_code=503, detail="Recommender index not loaded")

    try:
        # Build mask for optional filters
        mask = _apply_filters(CORPUS, q.split, q.label)

        # Build query vector
        if q.text is not None:
            qv = _vector_from_text(VECT, q.text)
            exclude_id = q.exclude_id
        else:
            qv = _vector_from_id(X_MAT, int(q.id))  # raises IndexError if out of range
            exclude_id = int(q.id) if q.exclude_id is None else int(q.exclude_id)

        # Compute top-K
        top = recommend_from_vector(
            qv=qv,
            X=X_MAT,
            corpus=CORPUS,
            top_k=q.k,
            mask=mask,
            exclude_id=exclude_id,
        )

        # Build response (truncate preview to ~300 chars)
        items: List[RecommendItem] = []
        for doc_id, score in top:
            split = str(CORPUS.loc[doc_id, "split"]) if "split" in CORPUS.columns else None
            label = int(CORPUS.loc[doc_id, "label"]) if "label" in CORPUS.columns else None
            text = str(CORPUS.loc[doc_id, "text"])
            preview = text[:300].replace("\n", " ")
            items.append(RecommendItem(id=int(doc_id), score=float(score), split=split, label=label, text=preview))

        return RecommendOut(results=items)

    except IndexError as e:
        raise HTTPException(status_code=400, detail=f"Invalid id: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommend error: {type(e).__name__}: {e}")
