"""
utils.py
========
Small helpers shared by both the training script and the Streamlit app, so that
metrics are computed in exactly the same way in both places.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET_COLUMN = "target"   # name of the label column in the CSV files

# Friendly display name -> saved model filename (used by the app's dropdown).
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def get_scores(pipeline, X):
    """Return a continuous score for the positive class, needed for the AUC.

    Prefer predict_proba; if the model does not expose it (e.g. a plain SVC),
    fall back to decision_function. AUC is rank-based, so either gives the same
    value for a given model.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def compute_metrics(pipeline, X, y):
    """Compute the six required metrics for a fitted pipeline on (X, y)."""
    y_pred = pipeline.predict(X)
    y_score = get_scores(pipeline, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
