import joblib
import csv
from pathlib import Path

# Path to the saved Logistic Regression + TF-IDF pipeline.
MODEL_PATH = Path("artifacts/sentiment_tfidf_logreg.joblib")
CSV_PATH = Path("artifacts/top_features.csv")

def main(top_k: int = 20):
    """
    Inspect the trained Logistic Regression model and print the top
    positive and negative features (words/phrases).
    Also saves the results into a CSV file for later analysis.
    
    Args:
        top_k (int): Number of features to display for each polarity.
    """
    # Ensure the trained model exists on disk before loading it.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.resolve()}. "
            f"Train it first: python -m src.models.train_baseline"
        )

    # Load the entire pipeline (TF-IDF vectorizer + Logistic Regression)
    pipe = joblib.load(MODEL_PATH)

    # Extract the TF-IDF vectorizer and the classifier from the pipeline.
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    # Get the feature names (words and n-grams) from the vectorizer.
    feature_names = vec.get_feature_names_out()

    # Logistic Regression stores one coefficient per feature.
    # For binary classification, coef_[0] corresponds to class "1" (positive).
    coefs = clf.coef_[0]

    # Select the top positive features: the largest coefficients.
    top_pos_idx = coefs.argsort()[-top_k:][::-1]

    # Select the top negative features: the smallest coefficients.
    top_neg_idx = coefs.argsort()[:top_k]

    # Print top features
    print(f"\nTop {top_k} POSITIVE features:")
    for i in top_pos_idx:
        print(f"{feature_names[i]:<25} {coefs[i]: .4f}")

    print(f"\nTop {top_k} NEGATIVE features:")
    for i in top_neg_idx:
        print(f"{feature_names[i]:<25} {coefs[i]: .4f}")

    # Save to CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "coefficient", "polarity"])
        for i in top_pos_idx:
            writer.writerow([feature_names[i], float(coefs[i]), "positive"])
        for i in top_neg_idx:
            writer.writerow([feature_names[i], float(coefs[i]), "negative"])

    print(f"\n✅ Top features saved to: {CSV_PATH}")

# Entry point: run the inspection when executing the script directly.
if __name__ == "__main__":
    # Default: show 25 features in each category.
    main(25)
