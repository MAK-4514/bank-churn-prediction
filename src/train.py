"""
train.py
--------
End-to-end training script for the Bank Customer Churn Prediction pipeline.

Workflow
--------
1. Load & split data
2. Build a ColumnTransformer preprocessing pipeline
3. Train LogisticRegression (baseline) and GradientBoostingClassifier (main)
4. Tune GBC with lightweight GridSearchCV
5. Optimise decision threshold (F1-optimal + Recall-priority)
6. Save fitted pipeline, metrics, and feature-importance artefacts
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend for saving figs
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    get_logger,
    ensure_dirs,
    load_data,
    save_metrics,
    NUM_FEATURES,
    CAT_FEATURES,
    TARGET,
    PROJECT_ROOT,
)

warnings.filterwarnings("ignore")

logger = get_logger("train")

RANDOM_STATE = 42          # keeps all stochastic operations reproducible
TEST_SIZE    = 0.20        # 80 / 20 train-test split


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing pipeline factory
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      • Numerical columns: fill missing values with the MEDIAN (robust to
        outliers common in financial data) then standardise via z-score.
      • Categorical columns: fill missing values with the MODE then apply
        OneHotEncoder with handle_unknown='ignore' so that unseen categories
        at inference time do not raise errors.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES),
        ],
        remainder="drop",   # silently ignore any unexpected columns
    )
    return preprocessor


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

def build_logistic_regression_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Logistic Regression with class_weight='balanced' to handle the typical
    ~20 % churn rate without manual over/undersampling.
    """
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", lr)])


def build_gbc_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """
    GradientBoostingClassifier as the primary non-linear model.
    Default hyper-parameters used as starting point for GridSearchCV.
    """
    gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("classifier", gbc)])


# ══════════════════════════════════════════════════════════════════════════════
# Threshold optimisation
# ══════════════════════════════════════════════════════════════════════════════

def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Iterate over all classification thresholds produced by roc_curve and return
    the one that maximises the F1 score on the validation/test set.
    Using the default 0.5 is a convenient convention but rarely optimal for
    imbalanced datasets; this finds the empirically best threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    logger.info(f"  Best F1 threshold: {best_thresh:.4f}  (F1={best_f1:.4f})")
    return float(best_thresh)


def find_threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.80
) -> tuple[float, float, float]:
    """
    Find the lowest threshold that achieves at least `target_recall`.
    Lower threshold => higher recall => lower precision.

    Returns
    -------
    (threshold, precision_at_threshold, achieved_recall)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_precision, best_recall = 0.5, 0.0, 0.0

    for t in sorted(thresholds, reverse=True):   # start high, work down
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall:
            prec = precision_score(y_true, y_pred, zero_division=0)
            best_thresh, best_precision, best_recall = t, prec, rec
            break   # take the highest threshold that still meets the target
    else:
        # Edge case: target recall unreachable, take lowest threshold
        t = thresholds[0]
        y_pred = (y_prob >= t).astype(int)
        best_thresh = t
        best_precision = precision_score(y_true, y_pred, zero_division=0)
        best_recall    = recall_score(y_true, y_pred, zero_division=0)

    logger.info(
        f"  Recall-priority threshold: {best_thresh:.4f}  "
        f"(Recall={best_recall:.4f}, Precision={best_precision:.4f})"
    )
    return float(best_thresh), float(best_precision), float(best_recall)


# ══════════════════════════════════════════════════════════════════════════════
# Feature-name extraction (post encoding)
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Reconstruct the full ordered list of feature names that the preprocessor
    outputs (numerical names + one-hot expanded categorical names).
    Required for interpretability plots.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    num_names = NUM_FEATURES.copy()

    cat_encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )
    cat_names = list(cat_encoder.get_feature_names_out(CAT_FEATURES))

    return num_names + cat_names


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "model",
    save_plots: bool = True,
    output_dir: Path = PROJECT_ROOT / "outputs",
) -> dict:
    """
    Compute full evaluation metrics and (optionally) save:
      • ROC curve figure
      • Confusion matrix figure

    Returns a dictionary of metric values.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_prob)

    logger.info(f"\n{'='*60}")
    logger.info(f"  Evaluation – {model_name}  (threshold={threshold:.4f})")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  F1        : {f1:.4f}")
    logger.info(f"  ROC-AUC   : {auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_roc_curve(y_test, y_prob, auc, model_name, output_dir)
        _plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

    return {
        "model":     model_name,
        "threshold": threshold,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
    }


def _plot_roc_curve(y_test, y_prob, auc, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#3b82f6", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve – {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_curve_{model_name.replace(' ','_')}.png", dpi=150)
    plt.close(fig)
    logger.info(f"  ROC curve saved → outputs/roc_curve_{model_name.replace(' ','_')}.png")


def _plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{model_name.replace(' ','_')}.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Confusion matrix saved → outputs/confusion_matrix_{model_name.replace(' ','_')}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Interpretability
# ══════════════════════════════════════════════════════════════════════════════

def extract_lr_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract and rank LogisticRegression coefficients.
    Positive coefficient → feature increases churn probability.
    Negative coefficient → feature decreases churn probability.
    """
    feature_names = get_feature_names(pipeline)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    df.sort_values("abs_coefficient", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_gbc_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract native feature importances from GradientBoostingClassifier
    (mean decrease in impurity, averaged over all trees).
    """
    feature_names = get_feature_names(pipeline)
    importances = pipeline.named_steps["classifier"].feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df.sort_values("importance", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def plot_feature_importance(df: pd.DataFrame, title: str, output_dir: Path, filename: str):
    """Horizontal bar chart of the top-15 features."""
    top_n = df.head(15).copy()
    col = "coefficient" if "coefficient" in df.columns else "importance"

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in top_n[col]]
    ax.barh(top_n["feature"][::-1], top_n[col][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)
    logger.info(f"  Feature importance chart saved → outputs/{filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Main training routine
# ══════════════════════════════════════════════════════════════════════════════

def train(data_path: str, drop_year: bool = True) -> None:
    ensure_dirs()
    output_dir = PROJECT_ROOT / "outputs"
    models_dir = PROJECT_ROOT / "models"

    # ── 1. Load data ────────────────────────────────────────────────────────
    logger.info(f"Loading data from: {data_path}")
    df = load_data(data_path, drop_year=drop_year)
    logger.info(f"  Shape: {df.shape}   Churn rate: {df[TARGET].mean()*100:.2f}%")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ── 2. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    preprocessor = build_preprocessor()

    # ══════════════════════════════════════════════════════════════════════════
    # A) Logistic Regression – baseline
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n─── Training Logistic Regression ───")
    lr_pipeline = build_logistic_regression_pipeline(build_preprocessor())
    lr_pipeline.fit(X_train, y_train)

    lr_prob  = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_thresh_f1           = find_best_threshold_f1(y_test.values, lr_prob)
    lr_thresh_recall, lr_prec_recall, lr_rec_recall = find_threshold_for_recall(
        y_test.values, lr_prob, target_recall=0.80
    )
    lr_metrics = evaluate_model(
        lr_pipeline, X_test, y_test,
        threshold=lr_thresh_f1,
        model_name="Logistic Regression",
        save_plots=True,
        output_dir=output_dir,
    )
    lr_metrics["threshold_f1_optimized"]     = lr_thresh_f1
    lr_metrics["threshold_recall_priority"]  = lr_thresh_recall
    lr_metrics["precision_at_recall_thresh"] = lr_prec_recall
    lr_metrics["recall_at_recall_thresh"]    = lr_rec_recall

    # LR interpretability
    lr_coef_df = extract_lr_coefficients(lr_pipeline)
    lr_coef_df.to_csv(output_dir / "lr_coefficients.csv", index=False)
    plot_feature_importance(
        lr_coef_df, "Logistic Regression – Coefficients (Top 15)",
        output_dir, "lr_coefficients.png"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # B) Gradient Boosting – main model with lightweight GridSearchCV
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n─── Tuning Gradient Boosting Classifier ───")
    gbc_pipeline = build_gbc_pipeline(build_preprocessor())

    # Small but meaningful grid; stratified 3-fold to respect class imbalance
    param_grid = {
        "classifier__n_estimators":  [100, 200],
        "classifier__max_depth":     [3, 4],
        "classifier__learning_rate": [0.05, 0.10],
        "classifier__subsample":     [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        gbc_pipeline,
        param_grid,
        scoring="roc_auc",     # optimise for discrimination ability
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_gbc_pipeline = grid_search.best_estimator_
    logger.info(f"  Best params: {grid_search.best_params_}")
    logger.info(f"  Best CV AUC: {grid_search.best_score_:.4f}")

    gbc_prob  = best_gbc_pipeline.predict_proba(X_test)[:, 1]
    gbc_thresh_f1          = find_best_threshold_f1(y_test.values, gbc_prob)
    gbc_thresh_recall, gbc_prec_recall, gbc_rec_recall = find_threshold_for_recall(
        y_test.values, gbc_prob, target_recall=0.80
    )
    gbc_metrics = evaluate_model(
        best_gbc_pipeline, X_test, y_test,
        threshold=gbc_thresh_f1,
        model_name="Gradient Boosting",
        save_plots=True,
        output_dir=output_dir,
    )
    gbc_metrics["threshold_f1_optimized"]     = gbc_thresh_f1
    gbc_metrics["threshold_recall_priority"]  = gbc_thresh_recall
    gbc_metrics["precision_at_recall_thresh"] = gbc_prec_recall
    gbc_metrics["recall_at_recall_thresh"]    = gbc_rec_recall

    # GBC interpretability
    gbc_fi_df = extract_gbc_feature_importance(best_gbc_pipeline)
    gbc_fi_df.to_csv(output_dir / "feature_importance.csv", index=False)
    plot_feature_importance(
        gbc_fi_df, "Gradient Boosting – Feature Importances (Top 15)",
        output_dir, "feature_importance.png"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # C) Save artefacts
    # ══════════════════════════════════════════════════════════════════════════

    # Primary model = GBC (better performance, same predict_proba interface)
    model_path = models_dir / "model.pkl"
    joblib.dump(best_gbc_pipeline, model_path)
    logger.info(f"\n  Primary model saved → {model_path}")

    # Also save LR for reference
    joblib.dump(lr_pipeline, models_dir / "lr_model.pkl")

    # Save chosen threshold alongside model so the Streamlit app can load it
    threshold_path = models_dir / "best_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "best_threshold_f1":           gbc_thresh_f1,
                "best_threshold_recall":       gbc_thresh_recall,
                "model":                       "GradientBoostingClassifier",
                "grid_search_best_params":     {
                    k: v for k, v in grid_search.best_params_.items()
                },
            },
            f,
            indent=4,
        )
    logger.info(f"  Threshold config saved → {threshold_path}")

    # Consolidated metrics
    all_metrics = {
        "logistic_regression":  lr_metrics,
        "gradient_boosting":    gbc_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    save_metrics(all_metrics, str(metrics_path))
    logger.info(f"  Metrics saved → {metrics_path}")

    logger.info("\n✅  Training complete. All artefacts written to models/ and outputs/")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train bank churn prediction models")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "European_Bank.csv"),
        help="Path to the CSV dataset",
    )
    parser.add_argument(
        "--keep-year",
        action="store_true",
        default=False,
        help="Keep the 'Year' column (dropped by default)",
    )
    args = parser.parse_args()
    train(data_path=args.data, drop_year=not args.keep_year)
