# ML Assignment — Binary Classification & Streamlit App (German Credit Risk)

An end-to-end machine-learning project that trains **five classification models** on the
UCI *Statlog German Credit* dataset, evaluates them with six metrics, and serves them through
an interactive **Streamlit** web app that scores an uploaded test CSV *without retraining*.

---

## a. Problem statement

Banks need to decide whether a loan applicant is a **good** or **bad** credit risk before
approving a loan. This is a **binary classification** problem: given an applicant's financial
and personal attributes, predict whether they are likely to default (**bad credit = 1**) or not
(**good credit = 0**). Automating this helps lenders reduce defaults while approving reliable
customers faster.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — *Statlog (German Credit Data)*
  (`https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/`)
- **Instances:** 1000 (≥ 500 required)
- **Features:** 20 (≥ 12 required) — 7 numeric (e.g. duration, credit amount, age) and
  13 categorical (e.g. checking-account status, credit history, purpose, employment, housing).
- **Target:** credit rating. Originally coded 1 = good, 2 = bad; we re-map it to
  **0 = good, 1 = bad**, so the positive class (`1`) is the risky "bad credit" group
  (300 of 1000 instances → a ~70/30 class imbalance).
- **Preprocessing:** numeric features are standardised (`StandardScaler`); categorical features
  are one-hot encoded (`OneHotEncoder`). Both steps are stored *inside* each saved model pipeline,
  so inference re-applies the exact same transformations.

## c. GitHub repository link

https://github.com/Zahirgit/german-credit-risk

## d. Models used and comparison

Five classifiers were trained on an identical 80/20 stratified train/test split
(`random_state=42`). All metrics below are computed on the **held-out test set** (200 rows) and
treat **class 1 (bad credit)** as the positive class.

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.750    | 0.8058 | 0.5581    | 0.8000 | 0.6575 | 0.4893 |
| Decision Tree          | 0.650    | 0.6741 | 0.4242    | 0.4667 | 0.4444 | 0.1903 |
| KNN                    | 0.755    | 0.7469 | 0.6667    | 0.3667 | 0.4731 | 0.3557 |
| Naive Bayes            | 0.695    | 0.7411 | 0.4937    | 0.6500 | 0.5612 | 0.3415 |
| Support Vector Machine | 0.760    | 0.8032 | 0.5750    | 0.7667 | 0.6571 | 0.4900 |

### Observations about model performance

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Best overall ranking (highest **AUC 0.806** and **F1 0.658**). Class balancing gives strong recall (0.80) on the bad-credit class, so it catches most risky applicants. The relationship in this data is largely linear, which suits it. |
| Decision Tree          | Weakest model (AUC 0.674, F1 0.444, MCC 0.190). A single depth-limited tree has high variance and ranks poorly, so its probability-based AUC suffers. |
| KNN                    | High precision (0.667) but poor recall (0.367) — it misses most bad-credit cases. Distance-based voting degrades in the high-dimensional sparse space created by one-hot encoding. |
| Naive Bayes            | Moderate, with decent recall (0.65) but low precision (0.49). Its feature-independence and Gaussian assumptions are violated by the correlated and one-hot-encoded credit features. |
| Support Vector Machine | Essentially tied for best: highest **Accuracy 0.760** and **MCC 0.490**, with strong recall (0.767) and AUC (0.803). The RBF kernel plus class balancing handle the data well. |
| **Overall winner for this dataset** | **Logistic Regression** — it has the best AUC and F1 and is the simplest, most interpretable model, with SVM a very close second. For credit scoring, Logistic Regression is also preferred because its coefficients are explainable to regulators. |

---

## Project structure

```
ML_Assigment_2/
│-- app.py              # Streamlit web app (loads saved models, scores uploaded CSV)
│-- train_models.py     # Trains & saves the 5 model pipelines + test_data.csv + metrics
│-- utils.py            # Shared metric helpers used by both app and training
│-- requirements.txt    # Dependencies (scikit-learn pinned for safe model loading)
│-- test_data.csv       # Held-out test data (features + target) used by the app
│-- README.md
│-- data/
│     └-- german.data   # Raw dataset (auto-downloaded by train_models.py if missing)
│-- model/
      │-- logistic_regression.joblib
      │-- decision_tree.joblib
      │-- knn.joblib
      │-- naive_bayes.joblib
      │-- svm.joblib
      └-- metrics.csv   # Saved comparison table
```

Each `.joblib` is a complete scikit-learn **Pipeline** (preprocessing + classifier), so the app
never has to retrain or re-fit any transformer.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # (optional) re-creates model/*.joblib and test_data.csv
streamlit run app.py
```

## Streamlit app features

- **CSV upload** — upload a test CSV (features + `target` column). If none is uploaded, the bundled
  `test_data.csv` is used.
- **Model selection dropdown** — pick any of the five trained models.
- **Evaluation metrics** — the six metrics for the selected model, plus a comparison table of all
  five models on the same data (best value per column highlighted).
- **Confusion matrix & classification report** — for the selected model.

## Live Streamlit app link

https://german-credit-risk-93qusfjhtzkyrxxpqcbeya.streamlit.app/
