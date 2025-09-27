"""
FastAPI application exposing two endpoints:

1. /predict → classify text as positive/negative
2. /recommend → retrieve top-K similar reviews
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.models.infer import predict
from src.recommender.recommend import recommend

app = FastAPI(title="EN Sentiment + Recommender")


class TextIn(BaseModel):
    text: str


class QueryIn(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def root():
    return {"ok": True, "service": "en-sentiment-recommender"}


@app.post("/predict")
def predict_endpoint(item: TextIn):
    label = predict(item.text)[0]
    return {"label": int(label), "label_name": "positive" if label == 1 else "negative"}


@app.post("/recommend")
def recommend_endpoint(q: QueryIn):
    res = recommend(q.query, q.top_k)
    return {
        "results": [
            {"rank": i + 1, "idx": idx, "score": score, "text": txt[:300]}
            for i, (idx, score, txt) in enumerate(res)
        ]
    }
