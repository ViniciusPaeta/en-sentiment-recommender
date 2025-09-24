# EN Sentiment + Content-based Recommender

AI project for **sentiment classification (EN/US)** and a **content-based recommender**.  
Built with **Python, scikit-learn, FastAPI, Streamlit**.

> Goal: start with a TF-IDF + Logistic Regression baseline (IMDB 50k) and a cosine-similarity recommender.  
> Later we can swap TF-IDF for sentence embeddings (Sentence-Transformers).

---

## 🔧 Tech stack
- Python, pandas, numpy, scikit-learn  
- FastAPI + Uvicorn (API)  
- Streamlit (demo UI)  
- (Optional) PyTorch + sentence-transformers  

---

## 📁 Project structure

```bash
en-sentiment-recommender/
├── artifacts/              # Saved models, metrics, plots
├── data/                   # Local CSV fallback (imdb_train.csv / imdb_test.csv)
├── notebooks/              # Jupyter notebooks (exploration/prototyping)
├── src/
│   ├── api/                # FastAPI backend (planned)
│   ├── app/                # Streamlit demo UI (planned)
│   ├── data/
│   │   └── load_imdb.py    # IMDB loader (Hugging Face + CSV fallback)
│   ├── models/
│   │   ├── train_baseline.py   # Train TF-IDF + Logistic Regression
│   │   ├── predict.py          # Predict texts from CLI
│   │   ├── inspect_baseline.py # Inspect top features
│   │   └── eval_baseline.py    # Evaluate model (ROC, Confusion Matrix, metrics)
│   └── utils/              # Helpers (future)
├── tests/                  # Unit tests (future)
├── Makefile                # Automation (train, eval, predict, inspect, clean)
├── requirements.txt        # Dependencies
└── README.md
```

---

## 📥 IMDB Data Loader

[`src/data/load_imdb.py`](src/data/load_imdb.py) provides a utility to load the [IMDB movie review dataset](https://huggingface.co/datasets/imdb).

### Features
- Loads IMDB from **Hugging Face Datasets** (`datasets.load_dataset("imdb")`).  
- Automatic **fallback to local CSV** (`data/imdb_train.csv` / `data/imdb_test.csv`).  
- Returns **pandas Series** with consistent dtypes (`string` for text, `Int8` for labels).  
- Validates input `split` and CSV schema.  

### Usage

```python
from src.data.load_imdb import load_data

X_train, y_train = load_data("train")
X_test, y_test = load_data("test")

print("Sample:", X_train.iloc[0], "Label:", y_train.iloc[0])
```

---

## 🤖 Baseline Model (TF-IDF + Logistic Regression)

Train and evaluate a sentiment classifier with TF-IDF features and Logistic Regression.

### Training

```bash
make train
```

- Default: full IMDB (25k train / 25k test).  
- Optional sampling for faster iterations:
  ```bash
  make train TRAIN_N=5000 TEST_N=2000
  ```

Artifacts saved in `artifacts/`:
- `sentiment_tfidf_logreg.joblib` → trained pipeline  
- `baseline_report.txt` → classification report  
- `metrics.json` → metrics (accuracy, etc.)  

### Evaluation

```bash
make eval
```

Generates:
- `eval_report.txt` (classification report, accuracy, ROC-AUC)  
- `metrics.json` (updated)  
- `roc_curve.png`  
- `confusion_matrix.png`  
- `confusion_matrix.csv`  

---

## 🔍 Model inspection & prediction

### Predict from CLI
```bash
make predict TEXT="this movie was surprisingly good!"
```

### Inspect top features
```bash
make inspect
```

Shows top positive and negative words/phrases learned by Logistic Regression.

---

## 📊 Results

Baseline results on the **IMDB 50k dataset**:

| Metric       | Value   |
|--------------|---------|
| Accuracy     | ~0.889  |
| Precision    | ~0.889  |
| Recall       | ~0.889  |
| F1-score     | ~0.889  |
| ROC-AUC      | ~0.95   |

Artifacts:
- `artifacts/eval_report.txt` → full classification report  
- `artifacts/roc_curve.png` → ROC curve  
- `artifacts/confusion_matrix.png` → confusion matrix  

Example ROC curve:  

![ROC Curve](artifacts/roc_curve.png)  

Example Confusion Matrix:  

![Confusion Matrix](artifacts/confusion_matrix.png)  

---

## ⚙️ Makefile automation

Main commands:

```bash
make train                # Train baseline
make eval                 # Evaluate baseline
make predict TEXT="..."   # Predict text
make inspect              # Inspect features
make clean                # Remove artifacts/
```

---

## 🚀 Next Steps / Roadmap

1. **Baseline Sentiment Classifier** ✅  
   - TF-IDF + Logistic Regression on IMDB.  

2. **Content-based Recommender**  
   - Cosine similarity between reviews.  

3. **Enhanced Representations**  
   - Replace TF-IDF with Sentence Embeddings (Sentence-Transformers).  

4. **API + Demo UI**  
   - FastAPI backend + Streamlit interface.  

5. **Deployment**  
   - Dockerize the service.  
   - Deploy on Heroku, Render, or AWS/GCP.  

---

✅ This project is now **end-to-end reproducible**: load → train → evaluate → inspect → predict.  
Next steps will expand to **recommender** and **deployment**.
