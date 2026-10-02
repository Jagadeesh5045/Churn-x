"""Train a customer churn classifier and produce evaluation + explainability artefacts.

Pipeline:
    data/churn.csv -> preprocessing (one-hot + passthrough) ->
    GradientBoostingClassifier -> metrics + permutation importance (+ SHAP if available)

Outputs:
    models/churn_model.pkl        fitted sklearn Pipeline (preprocess + model)
    models/feature_names.json     model input feature names (post-encoding)
    reports/metrics.json          test-set metrics + CV summary
    reports/confusion_matrix.png
    reports/permutation_importance.png / .csv
    reports/shap_summary.png      (only if the `shap` package is installed)
    reports/classification_report.txt

Usage:
    python train.py [--data data/churn.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "churn"
ID_COL = "customer_id"
RANDOM_STATE = 42

REPORTS = Path("reports")
MODELS = Path("models")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def encoded_feature_names(pipeline: Pipeline, numeric_features, categorical_features) -> list[str]:
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    return cat_names + list(numeric_features)


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Stayed", "Churned"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(y_true, y_proba, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend()
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_importance(result, feature_names: list[str], path: Path, top_n: int = 15) -> pd.DataFrame:
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color="#2c7bb6", ecolor="black", capsize=3)
    ax.set_xlabel("Mean decrease in ROC-AUC (permutation)")
    ax.set_title(f"Top {top_n} features — permutation importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return importances


def maybe_shap(pipeline: Pipeline, X_test: pd.DataFrame, feature_names: list[str], path: Path) -> bool:
    """Try a SHAP summary plot; return False if shap is unavailable."""
    try:
        import shap  # type: ignore
    except Exception:
        print("shap not installed — skipping SHAP summary (permutation importance covers explainability).")
        return False
    try:
        X_enc = pipeline.named_steps["preprocess"].transform(X_test)
        model = pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        sample = X_enc[:500]
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):  # binary-classification legacy output
            shap_values = shap_values[1]
        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary to {path}")
        return True
    except Exception as exc:  # never let SHAP break training
        print(f"SHAP failed ({exc}) — continuing without it.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    feature_cols = [c for c in df.columns if c not in (TARGET, ID_COL)]
    numeric_features = ["tenure", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
    numeric_features = [c for c in numeric_features if c in feature_cols]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X = df[feature_cols]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(numeric_features, categorical_features)

    # 5-fold stratified cross-validation (ROC-AUC) on the training set.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_overall": float(y.mean()),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "cv_auc_scores": [float(s) for s in cv_scores],
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, target_names=["Stayed", "Churned"])
    )

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), REPORTS / "confusion_matrix.png")
    plot_roc_pr_curves(y_test, y_proba, REPORTS / "roc_pr_curves.png")

    # Explainability: permutation importance on the test set.
    # NOTE: permutation_importance permutes the raw input columns of X_test
    # (pre-encoding), so importances map to the original feature names.
    perm = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
        scoring="roc_auc", n_jobs=-1,
    )
    importances = plot_permutation_importance(
        perm, feature_cols, REPORTS / "permutation_importance.png"
    )
    importances.to_csv(REPORTS / "permutation_importance.csv", index=False)

    # SHAP if available (optional) — needs the post-encoding feature names.
    feature_names = encoded_feature_names(pipeline, numeric_features, categorical_features)
    metrics["shap_available"] = maybe_shap(pipeline, X_test, feature_names, REPORTS / "shap_summary.png")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Persist artefacts for the dashboard.
    joblib.dump(pipeline, MODELS / "churn_model.pkl")
    (MODELS / "feature_names.json").write_text(json.dumps(
        {"features": feature_cols, "numeric": numeric_features, "categorical": categorical_features,
         "target": TARGET}, indent=2))

    print(json.dumps({k: v for k, v in metrics.items() if k != "cv_auc_scores"}, indent=2))
    print(f"\nArtefacts saved to {MODELS}/ and {REPORTS}/")


if __name__ == "__main__":
    main()
