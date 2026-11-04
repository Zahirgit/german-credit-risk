"""
app.py - Streamlit web app for ML Assignment 2
==============================================
Loads the five pre-trained classification pipelines and scores an uploaded test
CSV *without retraining*. It shows, for the chosen model, the six evaluation
metrics plus a confusion matrix and classification report, and it also shows a
comparison table of all five models on the same test data.

Run locally:   streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

from utils import MODEL_FILES, TARGET_COLUMN, compute_metrics

# Resolve paths relative to this file so the app works locally and when deployed.
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST = BASE_DIR / "test_data.csv"

CLASS_LABELS = ["Good credit (0)", "Bad credit (1)"]   # 1 = bad = positive class

st.set_page_config(page_title="German Credit - Classifier Comparison", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders (run once, then reused across interactions)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Load every saved pipeline into a dict {display_name: pipeline}."""
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_default_test():
    return pd.read_csv(DEFAULT_TEST)


def split_features_target(df):
    """Separate the feature columns from the target column."""
    if TARGET_COLUMN not in df.columns:
        return None, None
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


# ---------------------------------------------------------------------------
# Sidebar - controls
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")
st.sidebar.markdown(
    "Upload a **test CSV** (same feature columns as the training data plus a "
    "`target` column). If you don't upload anything, the bundled `test_data.csv` is used."
)
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
selected_model = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("German Credit Risk - Classifier Comparison")
st.write(
    "This app predicts whether a loan applicant is **good (0)** or **bad (1)** credit "
    "risk using five pre-trained models. Models are loaded from disk and only used "
    "for inference - they are **never retrained** on the uploaded data."
)

# Load the test data (uploaded or default) and the models.
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.success(f"Loaded uploaded file: {data.shape[0]} rows, {data.shape[1]} columns.")
else:
    data = load_default_test()
    st.info("Using the bundled `test_data.csv` (no file uploaded).")

models = load_all_models()
X, y = split_features_target(data)

if X is None:
    st.error(f"The CSV must contain a `{TARGET_COLUMN}` column (the true labels). "
             "Please upload a file that includes it.")
    st.stop()

# A small preview so the user can confirm the right file loaded.
with st.expander("Preview the test data"):
    st.dataframe(data.head())

# ---------------------------------------------------------------------------
# 1) Comparison table: all five models on this test data
# ---------------------------------------------------------------------------
st.header("All models - comparison on this test data")
try:
    comparison = pd.DataFrame(
        {name: compute_metrics(pipe, X, y) for name, pipe in models.items()}
    ).T.round(4)
except Exception as exc:  # e.g. uploaded CSV missing some feature columns
    st.error(f"Could not score the data. Make sure the CSV has the same feature "
             f"columns used in training.\n\nDetails: {exc}")
    st.stop()

# Highlight the best value in each metric column for readability.
st.dataframe(comparison.style.highlight_max(axis=0, color="#c6efce").format("{:.4f}"))

# ---------------------------------------------------------------------------
# 2) Details for the model chosen in the sidebar
# ---------------------------------------------------------------------------
st.header(f"Selected model: {selected_model}")
pipe = models[selected_model]
metrics = compute_metrics(pipe, X, y)

# Show the six metrics as compact cards.
cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, f"{value:.3f}")

y_pred = pipe.predict(X)

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, target_names=CLASS_LABELS,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

st.caption("Positive class = 1 (bad credit). Metrics are computed on the uploaded/bundled "
           "test data using the pre-trained saved models.")
