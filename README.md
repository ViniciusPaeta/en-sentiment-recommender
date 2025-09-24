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
en-sentiment-recommender/
├── data/                   # Local data (CSV fallback, e.g. imdb_train.csv / imdb_test.csv)
├── notebooks/              # Jupyter notebooks for exploration and prototyping
├── src/
│   ├── api/                # FastAPI backend
│   ├── app/                # Streamlit demo UI
│   ├── data/               # Data loaders and preprocessing
│   │   └── load_imdb.py    # IMDB loader (Hugging Face + CSV fallback)
│   ├── models/             # ML models, training scripts
│   └── utils/              # Helper functions (metrics, configs, etc.)
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── .gitignore

---

## 📥 IMDB Data Loader
We provide a small utility to load the [IMDB movie review dataset](https://huggingface.co/datasets/imdb).  
File: [`src/data/load_imdb.py`](src/data/load_imdb.py)

### Features
- Loads IMDB from **Hugging Face Datasets** (`datasets.load_dataset("imdb")`).
- Automatic **fallback to local CSV** (`data/imdb_train.csv` / `data/imdb_test.csv`) if Hugging Face is not available.
- Returns clean **pandas Series** with consistent dtypes (`string` for text, `Int8` for labels).
- Validates input `split` and CSV schema.

### Usage
python
from src.data.load_imdb import load_data

# Load training set
X_train, y_train = load_data("train")

# Load test set
X_test, y_test = load_data("test")

print("Sample review:", X_train.iloc[0])
print("Label:", y_train.iloc[0])  # 0 = negative, 1 = positive