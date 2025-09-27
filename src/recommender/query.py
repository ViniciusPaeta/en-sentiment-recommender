import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

RECO_DIR = Path("artifacts/recommender")
ROOT_DIR = Path("artifacts")

VECT_PATHS = [RECO_DIR / "tfidf_vectorizer.joblib", ROOT_DIR / "tfidf_vectorizer.joblib"]
MATRIX_PATHS = [RECO_DIR / "tfidf_matrix.npz", ROOT_DIR / "tfidf_matrix.npz"]
CORPUS_PATHS = [RECO_DIR / "corpus.csv", ROOT_DIR / "corpus.csv"]

def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {[str(p) for p in paths]}")

def load_index():
    vect_path = _first_existing(VECT_PATHS)
    matrix_path = _first_existing(MATRIX_PATHS)
    corpus_path = _first_existing(CORPUS_PATHS)
    vectorizer = joblib.load(vect_path)
    X = sparse.load_npz(matrix_path).tocsr()
    corpus = pd.read_csv(corpus_path)
    if "id" not in corpus.columns:
        corpus = corpus.reset_index().rename(columns={"index": "id"})
    if "text" not in corpus.columns:
        raise KeyError("corpus.csv must contain a 'text' column.")
    if "split" not in corpus.columns:
        corpus["split"] = "unknown"
    if "label" not in corpus.columns:
        corpus["label"] = -1
    return vectorizer, X, corpus

def _vector_from_text(vectorizer, text: str):
    q = vectorizer.transform([text])
    q = normalize(q, norm="l2", copy=False)
    return q.tocsr()

def _vector_from_id(X: sparse.csr_matrix, doc_id: int):
    if doc_id < 0 or doc_id >= X.shape[0]:
        raise IndexError(f"doc_id {doc_id} out of range [0, {X.shape[0]-1}]")
    return X.getrow(doc_id)

def _apply_filters(corpus: pd.DataFrame, split: Optional[str], label: Optional[int]) -> np.ndarray:
    mask = np.ones(len(corpus), dtype=bool)
    if split is not None:
        mask &= (corpus["split"].astype(str) == str(split))
    if label is not None:
        mask &= (corpus["label"].astype(int) == int(label))
    return mask

def recommend_from_vector(qv, X, corpus, top_k=5, mask=None, exclude_id=None) -> List[Tuple[int, float]]:
    if mask is None:
        mask = np.ones(X.shape[0], dtype=bool)
    if exclude_id is not None and 0 <= exclude_id < X.shape[0]:
        mask[exclude_id] = False
    idx_rows = np.where(mask)[0]
    X_sel = X[idx_rows, :]
    sims = (X_sel @ qv.T).toarray().ravel()
    if top_k <= 0 or top_k > sims.shape[0]:
        top_k = sims.shape[0]
    top_local = np.argpartition(-sims, top_k - 1)[:top_k]
    top_local = top_local[np.argsort(-sims[top_local])]
    top_global_idx = idx_rows[top_local]
    top_scores = sims[top_local]
    return list(zip(top_global_idx.tolist(), top_scores.tolist()))

def format_results(results, corpus, show_text=True) -> str:
    lines = []
    header = f"{'rank':<4} {'id':<8} {'score':<8} {'split':<8} {'label':<5} {'text' if show_text else ''}"
    lines.append(header.rstrip())
    lines.append("-" * len(header))
    for r, (doc_id, score) in enumerate(results, start=1):
        split = str(corpus.loc[doc_id, "split"]) if "split" in corpus.columns else "na"
        label = str(corpus.loc[doc_id, "label"]) if "label" in corpus.columns else "na"
        if show_text:
            text = str(corpus.loc[doc_id, "text"])[:160].replace("\n", " ")
            lines.append(f"{r:<4} {doc_id:<8} {score:<8.4f} {split:<8} {label:<5} {text}")
        else:
            lines.append(f"{r:<4} {doc_id:<8} {score:<8.4f} {split:<8} {label:<5}")
    return "\n".join(lines)

def save_results_csv(out_path: Path, results, corpus) -> None:
    rows = []
    for doc_id, score in results:
        row = {
            "id": int(doc_id),
            "score": float(score),
            "split": corpus.loc[doc_id, "split"] if "split" in corpus.columns else None,
            "label": corpus.loc[doc_id, "label"] if "label" in corpus.columns else None,
            "text": corpus.loc[doc_id, "text"],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Query TF-IDF cosine index.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str)
    g.add_argument("--id", type=int)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--label", type=int, default=None)
    p.add_argument("--exclude-id", type=int, default=None)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--out-csv", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    vectorizer, X, corpus = load_index()
    mask = _apply_filters(corpus, args.split, args.label)
    if args.text is not None:
        qv = _vector_from_text(vectorizer, args.text)
        exclude_id = args.exclude_id
    else:
        qv = _vector_from_id(X, int(args.id))
        exclude_id = int(args.id) if args.exclude_id is None else int(args.exclude_id)
    results = recommend_from_vector(qv, X, corpus, top_k=args.k, mask=mask, exclude_id=exclude_id)
    print(format_results(results, corpus, show_text=not args.no_text))
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_csv(out_path, results, corpus)
        print(f"\n✅ Results saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
