from typing import Tuple
from pathlib import Path
import warnings
import pandas as pd


def load_data(split: str = "train") -> Tuple[pd.Series, pd.Series]:
    """
    Load the IMDB dataset (movie reviews) as two pandas Series: (X, y).

    Strategy:
      1) Try to load via Hugging Face Datasets (`load_dataset("imdb")`).
      2) If not possible (e.g., `datasets` not installed, no internet, cache error),
         fallback to a local CSV file at `data/imdb_{split}.csv`.

    Parameters
    ----------
    split : {"train", "test"}, optional (default="train")
        Which partition of the dataset to load.

    Returns
    -------
    (X, y) : Tuple[pd.Series, pd.Series]
        X = texts (dtype pandas "string"), y = binary labels (dtype "Int8": 0=neg, 1=pos).

    Explicit errors (for clarity and debugging)
    -------------------------------------------
    - ValueError: if `split` is invalid.
    - FileNotFoundError: if the fallback is used and CSV is missing.
    - KeyError: if the CSV does not contain required columns ("text" and "label").
    """

    # ===== 0) Validate the `split` argument =====
    valid_splits = {"train", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split!r}. Use 'train' or 'test'.")

    # ===== 1) Try Hugging Face Datasets (lazy import) =====
    # Import is done inside the function to avoid hard dependency
    # if you only want to work with local CSVs.
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        # If `datasets` is not installed, go directly to CSV fallback.
        return _load_from_csv(split)

    # ===== 2) Load via HF Datasets with error handling =====
    try:
        # Download/cache the IMDB dataset.
        ds = load_dataset("imdb")  # returns DatasetDict with "train" and "test"

        # Select the requested split explicitly.
        part = ds[split]

        # Convert columns to pandas Series with consistent dtypes:
        # - texts as pandas "string" (better than "object" for NLP),
        # - labels as "Int8" (sufficient for 0/1 and memory efficient).
        X = pd.Series(part["text"], dtype="string")
        y = pd.Series(part["label"], dtype="Int8")

        # reset_index(drop=True) ensures a clean sequential index (0..n-1).
        return X.reset_index(drop=True), y.reset_index(drop=True)

    except Exception as e:
        # Any unexpected error in HF Datasets loading (network, cache, etc.)
        # triggers fallback to CSV, with a warning to make it visible.
        warnings.warn(
            f"Failed to load via Hugging Face Datasets "
            f"({type(e).__name__}: {e}). Falling back to local CSV.",
            RuntimeWarning,
        )
        return _load_from_csv(split)


def _load_from_csv(split: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load IMDB data from local CSV file `data/imdb_{split}.csv`, validating schema.

    The CSV must contain two columns:
      - "text": free-form review text (will be converted to dtype 'string'),
      - "label": binary label 0/1 (will be converted to dtype 'Int8').

    Parameters
    ----------
    split : {"train", "test"}
        Partition to load, used to build the filename.

    Returns
    -------
    (X, y) : Tuple[pd.Series, pd.Series]
        Series with consistent dtypes and reset index.
    """
    # Build the expected path based on split.
    path = Path("data") / f"imdb_{split}.csv"

    # Early check: does the file exist?
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    # Read the CSV into pandas DataFrame.
    # Add options here if needed (e.g., encoding="utf-8", on_bad_lines="skip").
    df = pd.read_csv(path)

    # Validate schema: required columns must be present.
    required_cols = {"text", "label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(
            f"CSV {path} must contain columns {sorted(required_cols)}; "
            f"missing {sorted(missing)}."
        )

    # Convert to proper dtypes: text -> "string", label -> "Int8".
    X = df["text"].astype("string")
    y = df["label"].astype("Int8")

    # Reset index to sequential 0..n-1 before returning.
    return X.reset_index(drop=True), y.reset_index(drop=True)


# Quick usage example:
# if __name__ == "__main__":
#     X_train, y_train = load_data("train")
#     print(X_train.head(), y_train.value_counts(dropna=False))
