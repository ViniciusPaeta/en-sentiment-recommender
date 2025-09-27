"""
IMDB DATA LOADER (highly commented)

Goal
----
Provide a single function `load_data(split)` that returns (X, y) as pandas Series.
We first try to pull the "imdb" dataset from HuggingFace `datasets` (auto-download,
local cache, easy to use). If that fails (e.g., offline), we fall back to CSVs
under `data/` following the naming convention: data/imdb_train.csv, data/imdb_test.csv.

Expected CSV schema
-------------------
- Column 'text'  : raw review text (string)
- Column 'label' : integer class (0 = negative, 1 = positive)

Why pandas Series?
------------------
Keeping X and y as Series is convenient for sklearn and preserves indexing semantics.
"""

from typing import Any  # noqa: F401 (kept in case you want to extend types)
import pandas as pd


def load_data(split: str = "train") -> tuple[pd.Series, pd.Series]:
    """
    Load texts (X) and labels (y) for the given split.

    Parameters
    ----------
    split : str
        Must be "train" or "test".

    Returns
    -------
    (X, y) : tuple[pd.Series, pd.Series]
        X contains review strings; y contains integer labels (0/1).
    """
    try:
        # PRIMARY PATH: HuggingFace datasets (zero setup, cached)
        from datasets import load_dataset

        ds = load_dataset("imdb")
        part = ds["train"] if split == "train" else ds["test"]
        X = pd.Series(part["text"])
        y = pd.Series(part["label"])
        return X, y
    except Exception:
        # FALLBACK PATH: local CSVs (useful offline or for custom subsets)
        df = pd.read_csv(f"data/imdb_{split}.csv")
        return df["text"], df["label"]
