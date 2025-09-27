# EN Sentiment + Content-Based Recommender

AI project for **sentiment classification (EN/US)** and a **content-based recommender**.  
Baseline: **TF-IDF + Logistic Regression** for sentiment and **TF-IDF + cosine similarity** for recommendation.  
Backend built with **FastAPI** and demo UI with **Streamlit**.

---

## Features
- Sentiment classifier (TF-IDF + Logistic Regression)
- Content-based recommender (TF-IDF + cosine similarity)
- FastAPI service (/predict, /recommend)
- Streamlit demo UI
- Clean repository (artifacts ignored)
- Basic CI and pre-commit hooks

---

## Tech Stack
- Python, pandas, numpy, scikit-learn, scipy, joblib
- FastAPI + Uvicorn
- Streamlit
- HuggingFace `datasets` (IMDB 50k)

---

## Project Layout

```bash
.
├── src/
│   ├── api/                # FastAPI app (endpoints)
│   ├── app/                # Streamlit UI (manual testing)
│   ├── data/               # Data loader (HuggingFace or CSV fallback)
│   ├── models/             # Training + inference
│   └── recommender/        # TF-IDF index + cosine recommender
├── tests/                  # Smoke tests / future unit tests
├── artifacts/              # (ignored) models and indices generated locally
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
└── README.md
```

> The `artifacts/` directory is ignored: models and indices are generated locally.

---

## Quickstart

### 1) Environment
```bash
pip install -r requirements.txt
```

### 2) Train the baseline (TF-IDF + Logistic Regression)
```bash
python -m src.models.train_baseline
# -> artifacts/sentiment_tfidf_logreg.joblib
```

### 3) Build the recommender index (TF-IDF + cosine)
```bash
python -m src.recommender.build_index
# -> artifacts/tfidf_vectorizer.joblib, corpus.joblib, tfidf_matrix.npz
```

### 4) Run the API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Endpoints
- GET `/` → health check  
- POST `/predict`  
  ```json
  {"text": "I loved this movie!"}
  ```
  →  
  ```json
  {"label": 1, "label_name": "positive"}
  ```
- POST `/recommend`  
  ```json
  {"query": "space adventure with friendship", "top_k": 3}
  ```
  →  
  ```json
  {"results":[{"rank":1,"idx":123,"score":0.71,"text":"..."}]}
  ```

### 5) Run the UI (Streamlit)
```bash
export PYTHONPATH=$PWD
streamlit run src/app/streamlit_app.py
# open http://localhost:8501
```

---

## Tests
```bash
pytest -q
```

---

## Development

### Pre-commit
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Lint / Format / Types
```bash
ruff check .
black .
mypy src
```

---

## CSV Fallback (optional)
If offline or you prefer CSVs, create:
- `data/imdb_train.csv`
- `data/imdb_test.csv`

With columns:
- `text` (string)
- `label` (0 = negative, 1 = positive)

The loader tries HuggingFace first and falls back to CSVs.

---

## Troubleshooting

**ModuleNotFoundError: src**  
```bash
export PYTHONPATH=$PWD
```

**API not responding**  
Ensure Uvicorn is running:
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Model or index not found**  
Run:
```bash
python -m src.models.train_baseline
python -m src.recommender.build_index
```

**Pretty JSON in terminal**  
On Arch:
```bash
sudo pacman -S jq
```

---

## Roadmap
- Replace TF-IDF with embeddings (`sentence-transformers`)
- Add probabilities in sentiment output
- Dockerfile for API/UI
- DVC for dataset/model tracking

---

## License
MIT — see LICENSE.

