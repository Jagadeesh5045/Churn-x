"""Streamlit dashboard for the churn-x customer churn project.

Sections:
    1. KPI cards — churn rate, test AUC, test F1, test recall
    2. What-if predictor — tune a customer's features, get churn probability
       plus the top factors driving that prediction
    3. Exploratory charts — churn distribution slices
    4. Model explainability — permutation importance + confusion matrix

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "churn_model.pkl"
METRICS_PATH = BASE / "reports" / "metrics.json"
IMPORTANCES_PATH = BASE / "reports" / "permutation_importance.csv"
DATA_PATH = BASE / "data" / "churn.csv"
NAMES_PATH = BASE / "models" / "feature_names.json"


@st.cache_resource
def load_artefacts():
    import json

    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python data/generate_data.py` then `python train.py` first.")
        st.stop()
    pipeline = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    names = json.loads(NAMES_PATH.read_text())
    df = pd.read_csv(DATA_PATH)
    importances = pd.read_csv(IMPORTANCES_PATH)
    return pipeline, metrics, names, df, importances


CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["Yes", "No", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "online_backup": ["Yes", "No", "No internet service"],
    "device_protection": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def local_drivers(pipeline, row: pd.DataFrame, baseline: dict, top_n: int = 5) -> pd.DataFrame:
    """Per-customer drivers via single-feature replacement against a baseline.

    For each feature, replace the customer's value with the baseline
    (median/mode) and measure the change in churn probability. A positive
    delta means the customer's value pushes them TOWARDS churn.
    """
    base_proba = float(pipeline.predict_proba(row)[0, 1])
    deltas = []
    for col in row.columns:
        perturbed = row.copy()
        perturbed[col] = baseline[col]
        p = float(pipeline.predict_proba(perturbed)[0, 1])
        deltas.append({"feature": col, "delta": base_proba - p,
                       "customer_value": row[col].iloc[0]})
    out = pd.DataFrame(deltas).sort_values("delta", ascending=False)
    return out.head(top_n), base_proba


def main() -> None:
    st.set_page_config(page_title="churn-x · Customer Churn Dashboard", layout="wide")
    pipeline, metrics, names, df, importances = load_artefacts()

    st.title("📊 churn-x — Customer Churn Prediction & Explainability")
    st.caption("Gradient Boosting classifier on synthetic telecom data · "
               "permutation-importance explainability · what-if simulator")

    # ---------------- KPI cards ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn rate (dataset)", f"{metrics['churn_rate_overall']:.1%}")
    c2.metric("Test ROC-AUC", f"{metrics['test_auc']:.3f}",
              delta=f"CV {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    c3.metric("Test F1", f"{metrics['test_f1']:.3f}")
    c4.metric("Test precision / recall",
              f"{metrics['test_precision']:.3f} / {metrics['test_recall']:.3f}")

    st.divider()

    # ---------------- What-if predictor ----------------
    st.header("🔮 What-if churn predictor")
    st.write("Tune a customer's profile and see the predicted churn probability "
             "plus the top factors driving it.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Customer profile")
        inputs: dict = {}
        inputs["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
        inputs["monthly_charges"] = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=1.0)
        # Total charges is derived, as in real billing: tenure x monthly charges.
        inputs["total_charges"] = round(inputs["tenure"] * inputs["monthly_charges"], 2)
        st.caption(f"Total charges (auto): ${inputs['total_charges']:,.2f}")
        inputs["support_calls"] = st.slider("Support calls (last 6 months)", 0, 10, 2)
        inputs["senior_citizen"] = st.selectbox("Senior citizen", [0, 1],
                                                format_func=lambda v: "Yes" if v else "No")
        inputs["contract"] = st.selectbox("Contract", CATEGORICAL_OPTIONS["contract"])
        inputs["internet_service"] = st.selectbox("Internet service", CATEGORICAL_OPTIONS["internet_service"])
        inputs["tech_support"] = st.selectbox("Tech support", CATEGORICAL_OPTIONS["tech_support"])
        inputs["payment_method"] = st.selectbox("Payment method", CATEGORICAL_OPTIONS["payment_method"])
        inputs["paperless_billing"] = st.selectbox("Paperless billing", ["Yes", "No"])
        # Sensible defaults for the remaining features (most common values).
        inputs["gender"] = st.selectbox("Gender", ["Male", "Female"])
        inputs["partner"] = st.selectbox("Partner", ["Yes", "No"])
        inputs["dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        inputs["phone_service"] = "Yes"
        inputs["multiple_lines"] = "No"
        inputs["online_security"] = "No"
        inputs["online_backup"] = "No"
        inputs["device_protection"] = "No"
        inputs["streaming_tv"] = "No"
        inputs["streaming_movies"] = "No"

    feature_cols = names["features"]
    row = pd.DataFrame([{c: inputs[c] for c in feature_cols}])

    baseline = {}
    for c in feature_cols:
        baseline[c] = df[c].median() if c in names["numeric"] else df[c].mode().iloc[0]

    drivers, proba = local_drivers(pipeline, row, baseline)

    with right:
        st.subheader("Prediction")
        st.metric("Churn probability", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
        risk = "🔴 High risk" if proba >= 0.6 else "🟡 Medium risk" if proba >= 0.35 else "🟢 Low risk"
        st.write(f"**{risk}** of churning.")

        st.subheader("Top driving factors")
        st.caption("How each feature value moves this customer's risk vs. a typical customer. "
                   "Red pushes towards churn, green away from it.")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        colors = ["#d73027" if d >= 0 else "#1a9850" for d in drivers["delta"]]
        labels = [f"{f} = {v}" for f, v in zip(drivers["feature"], drivers["customer_value"])]
        ax.barh(labels[::-1], drivers["delta"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Change in churn probability")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    st.divider()

    # ---------------- Exploratory charts ----------------
    st.header("📈 Churn in the data")
    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Churn rate by contract type")
        by_contract = df.groupby("contract")["churn"].mean().reindex(
            ["Month-to-month", "One year", "Two year"])
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.bar(by_contract.index, by_contract.values, color=["#d73027", "#fdae61", "#1a9850"])
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(by_contract.values) * 1.25)
        for x, v in zip(by_contract.index, by_contract.values):
            ax.text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with e2:
        st.subheader("Monthly charges distribution by churn")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(df.loc[df.churn == 0, "monthly_charges"], bins=30, alpha=0.6, label="Stayed")
        ax.hist(df.loc[df.churn == 1, "monthly_charges"], bins=30, alpha=0.6, label="Churned")
        ax.set_xlabel("Monthly charges ($)")
        ax.set_ylabel("Customers")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    e3, e4 = st.columns(2)
    with e3:
        st.subheader("Churn rate by number of support calls")
        by_calls = df.groupby("support_calls")["churn"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(by_calls.index, by_calls.values, marker="o", color="#2c7bb6")
        ax.set_xlabel("Support calls (last 6 months)")
        ax.set_ylabel("Churn rate")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with e4:
        st.subheader("Tenure distribution by churn")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(df.loc[df.churn == 0, "tenure"], bins=24, alpha=0.6, label="Stayed")
        ax.hist(df.loc[df.churn == 1, "tenure"], bins=24, alpha=0.6, label="Churned")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Customers")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    st.divider()

    # ---------------- Explainability ----------------
    st.header("🧠 Model explainability")
    x1, x2 = st.columns(2)
    with x1:
        st.subheader("Permutation importance (top 15)")
        top = importances.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
                color="#2c7bb6", ecolor="black", capsize=3)
        ax.set_xlabel("Mean decrease in ROC-AUC")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with x2:
        st.subheader("Confusion matrix (test set)")
        cm_path = BASE / "reports" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), width="stretch")
        cm = np.array(metrics["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        st.write(f"True negatives: **{tn:,}** · False positives: **{fp:,}** · "
                 f"False negatives: **{fn:,}** · True positives: **{tp:,}**")
        shap_path = BASE / "reports" / "shap_summary.png"
        if shap_path.exists():
            st.subheader("SHAP summary")
            st.image(str(shap_path), width="stretch")

    with st.expander("Classification report (test set)"):
        st.code((BASE / "reports" / "classification_report.txt").read_text())

    st.caption("Built with scikit-learn · Streamlit · matplotlib. Synthetic data — no real customers.")


if __name__ == "__main__":
    main()
