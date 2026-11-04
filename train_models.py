"""
train_models.py
================
Trains five classification models on the UCI Statlog German Credit dataset and
saves each one (together with its preprocessing steps) as a .joblib pipeline.

Why a full scikit-learn *Pipeline* per model?
    The Streamlit app must load a saved model and score a freshly uploaded test
    CSV *without retraining*. By saving the preprocessing (scaling + one-hot
    encoding) and the classifier together in one Pipeline, the app can simply call
    `pipeline.predict(raw_features)` - the exact same transformations used in
    training are re-applied automatically, so there is no train/serve skew.

Run:  python train_models.py
Output: model/*.joblib  (five pipelines),  model/metrics.csv,  test_data.csv
"""
import os
import urllib.request

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# The five required classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils import compute_metrics   # shared metric helper (same code the app uses)

RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_PATH = os.path.join("data", "german.data")

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
# The raw file has no header and is space-separated. We give every column a
# readable name taken from the UCI dataset documentation. 13 columns are
# categorical (coded like "A11", "A34") and 7 are numeric.
COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property_magnitude", "age",
    "other_installment_plans", "housing", "existing_credits", "job",
    "num_liable", "telephone", "foreign_worker", "target",
]
NUMERIC = ["duration", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "num_liable"]
CATEGORICAL = [c for c in COLUMNS if c not in NUMERIC and c != "target"]


def load_data():
    """Download the dataset once, then read it into a DataFrame."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=COLUMNS)
    # Target is originally 1 = good credit, 2 = bad credit. We re-map it so that
    # 1 = "bad credit" (the risky, minority class we care about) and 0 = "good".
    df["target"] = (df["target"] == 2).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Build the shared preprocessing step
# ---------------------------------------------------------------------------
def build_preprocessor():
    """Scale numeric columns and one-hot encode categorical columns.

    `handle_unknown="ignore"` keeps the app robust: if an uploaded CSV happens to
    contain a category value not seen during training, it is encoded as all-zeros
    instead of raising an error.
    """
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


# ---------------------------------------------------------------------------
# 3. Define the five models
# ---------------------------------------------------------------------------
def build_models():
    """Return a dict of {display_name: (filename, estimator)}.

    Notes on a few choices:
      - LogisticRegression / SVC use class_weight="balanced" because the classes
        are imbalanced (700 good vs 300 bad); this helps recall on the bad class.
      - For SVC we use the default decision_function for the AUC (see utils.get_scores),
        so we avoid the deprecated probability=True parameter.
    """
    return {
        "Logistic Regression": ("logistic_regression.joblib",
                                 LogisticRegression(max_iter=1000, class_weight="balanced",
                                                    random_state=RANDOM_STATE)),
        "Decision Tree": ("decision_tree.joblib",
                          DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "KNN": ("knn.joblib", KNeighborsClassifier(n_neighbors=15)),
        "Naive Bayes": ("naive_bayes.joblib", GaussianNB()),
        "Support Vector Machine": ("svm.joblib",
                                   SVC(class_weight="balanced", random_state=RANDOM_STATE)),
    }


def main():
    df = load_data()
    X = df.drop(columns="target")
    y = df["target"]

    # Hold out 20% as the test set (stratified to keep the 70/30 class ratio).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    # Save the test set exactly as the app will receive it: raw features + target.
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv  ({test_df.shape[0]} rows, {X_test.shape[1]} features)")

    os.makedirs("model", exist_ok=True)
    results = {}
    for name, (filename, estimator) in build_models().items():
        # One pipeline = preprocessing + classifier, trained together.
        pipe = Pipeline([("preprocess", build_preprocessor()), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, os.path.join("model", filename))   # persist for the app
        results[name] = compute_metrics(pipe, X_test, y_test)
        print(f"Trained and saved: {name}")

    # Comparison table for the README.
    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "ML Model"
    metrics_df.to_csv(os.path.join("model", "metrics.csv"))
    print("\n=== Evaluation metrics on the held-out test set ===")
    print(metrics_df)
    print("\nBest model by F1:", metrics_df["F1"].idxmax(),
          "| by AUC:", metrics_df["AUC"].idxmax())


if __name__ == "__main__":
    main()
