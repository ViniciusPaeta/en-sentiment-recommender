import os
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from src.data.load_imdb import load_data


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "sentiment_tfidf_logreg.joblib"
REPORT_PATH = ARTIFACTS_DIR / "eval_report.txt"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
ROC_PNG = ARTIFACTS_DIR / "roc_curve.png"
CM_PNG = ARTIFACTS_DIR / "confusion_matrix.png"
CM_CSV = ARTIFACTS_DIR / "confusion_matrix.csv"


def _update_metrics(path: Path, new_metrics: Dict[str, Any]) -> None:
    """
    Merge new_metrics into an existing JSON (if present) to keep a single source of truth.
    """
    data = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # If parsing fails, start fresh (avoid crashing the eval step).
            data = {}
    data.update(new_metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _ensure_model() -> Any:
    """
    Load the trained pipeline (TF-IDF + LogisticRegression).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.resolve()}. "
            f"Train it first: python -m src.models.train_baseline"
        )
    return joblib.load(MODEL_PATH)


def _maybe_sample(X, y, n):
    """
    Optional fast evaluation by subsampling N examples.
    No stratification (simple & fast), fixed RNG for reproducibility.
    """
    if not n or n <= 0 or n >= len(X):
        return X, y
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=int(n), replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> None:
    """
    Draw ROC curve and save to disk.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Baseline TF-IDF + LogReg")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path, labels=("neg(0)", "pos(1)")) -> None:
    """
    Draw Confusion Matrix heatmap and save to disk.
    """
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix — Baseline TF-IDF + LogReg")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    """
    Evaluate baseline model on IMDB test split:
    - Loads model and test data
    - Computes accuracy, ROC-AUC, classification report and confusion matrix
    - Saves plots (ROC, CM) and metrics (txt + JSON)
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and data
    pipe = _ensure_model()
    X_test, y_test = load_data("test")

    # Optional sampling to speed up eval (env var)
    test_sample = int(os.getenv("BASELINE_TEST_SAMPLE", "0"))
    X_test, y_test = _maybe_sample(X_test, y_test, test_sample)

    # Predictions: proba -> for ROC/AUC; class -> for confusion matrix/report
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        # Fall back to decision_function or raw predictions if necessary
        if hasattr(pipe, "decision_function"):
            # Decision function can be converted to pseudo-probabilities via ranking,
            # but we simply use the raw scores for ROC-AUC (acceptable for comparison).
            y_score = pipe.decision_function(X_test)
        else:
            # As a last resort, use predicted labels (not ideal for ROC).
            y_score = pipe.predict(X_test)

    y_pred = pipe.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Print to console
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    # Save text report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + f"\nAccuracy: {acc:.4f}\nROC-AUC:  {auc:.4f}\n")

    # Save confusion matrix as CSV (for quick inspection in spreadsheets)
    np.savetxt(CM_CSV, cm, fmt="%d", delimiter=",")
    # Save plots
    _plot_roc_curve(np.asarray(y_test), np.asarray(y_score), ROC_PNG)
    _plot_confusion_matrix(cm, CM_PNG, labels=("neg(0)", "pos(1)"))

    # Save/update JSON metrics
    _update_metrics(
        METRICS_JSON,
        {
            "accuracy": float(acc),
            "roc_auc": float(auc) if not np.isnan(auc) else None,
            "n_test": int(len(X_test)),
        },
    )

    print(f"📝 Eval report: {REPORT_PATH}")
    print(f"📊 ROC curve:  {ROC_PNG}")
    print(f"🧮 Confusion:  {CM_PNG} (and CSV at {CM_CSV})")


if __name__ == "__main__":
    main()
