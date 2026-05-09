"""
evaluate.py
-----------
Stand-alone evaluation script.
Loads a saved pipeline from models/model.pkl, evaluates it on the
held-out test set (re-created by using the SAME random_state=42 split),
and re-generates all evaluation plots and metrics.

Usage
-----
python src/evaluate.py --data data/European_Bank.csv
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    get_logger,
    load_data,
    save_metrics,
    assign_risk_band,
    assign_risk_band_optimized,
    TARGET,
    PROJECT_ROOT,
)

warnings.filterwarnings("ignore")
logger = get_logger("evaluate")

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ══════════════════════════════════════════════════════════════════════════════

def full_evaluation(data_path: str) -> None:
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models"

    # ── Load pipeline & threshold config ────────────────────────────────────
    model_path     = models_dir / "model.pkl"
    threshold_path = models_dir / "best_threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. Run train.py first."
        )

    pipeline = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    with open(threshold_path) as f:
        threshold_cfg = json.load(f)
    best_thresh_f1     = threshold_cfg["best_threshold_f1"]
    best_thresh_recall = threshold_cfg["best_threshold_recall"]
    logger.info(
        f"  F1-optimal threshold:     {best_thresh_f1:.4f}\n"
        f"  Recall-priority threshold:{best_thresh_recall:.4f}"
    )

    # ── Reproduce the same test split (MUST match train.py) ─────────────────
    df = load_data(data_path)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Test set size: {len(X_test)} rows")

    # ── Predict ──────────────────────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ── Threshold comparison table ───────────────────────────────────────────
    logger.info("\n── Threshold Comparison ──────────────────────────────────────")
    thresholds_to_compare = {
        "Default (0.50)":    0.50,
        "F1-optimal":        best_thresh_f1,
        "Recall-priority":   best_thresh_recall,
    }
    rows = []
    for name, t in thresholds_to_compare.items():
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "Threshold Label": name,
            "Threshold Value": round(t, 4),
            "Accuracy":        round(accuracy_score(y_test, y_pred), 4),
            "Precision":       round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":          round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":              round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":         round(roc_auc_score(y_test, y_prob), 4),
        })
    comparison_df = pd.DataFrame(rows)
    logger.info("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv(output_dir / "threshold_comparison.csv", index=False)

    # ── Full classification report at best F1 threshold ──────────────────────
    y_pred_best = (y_prob >= best_thresh_f1).astype(int)
    logger.info(
        "\n── Classification Report (F1-optimal threshold) ──\n"
        + classification_report(y_test, y_pred_best, target_names=["Retained", "Churned"])
    )

    # ── Risk scoring sample ──────────────────────────────────────────────────
    risk_default  = assign_risk_band(y_prob)
    risk_optimized = assign_risk_band_optimized(y_prob, opt_threshold=best_thresh_f1)

    risk_df = X_test.copy()
    risk_df["churn_probability"] = np.round(y_prob, 4)
    risk_df["risk_band_default"]  = risk_default
    risk_df["risk_band_optimized"] = risk_optimized
    risk_df["actual_churn"]       = y_test.values
    risk_df["predicted_churn_f1"] = y_pred_best

    risk_df.to_csv(output_dir / "risk_scores.csv", index=False)
    logger.info(f"  Risk scores saved → outputs/risk_scores.csv  ({len(risk_df)} rows)")

    # Risk band distribution
    logger.info("\n── Risk Band Distribution (default bands) ──")
    band_counts = risk_df["risk_band_default"].value_counts()
    logger.info(band_counts.to_string())

    # ── Precision-Recall curve ───────────────────────────────────────────────
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_arr, prec_arr, lw=2, color="#8b5cf6")
    ax.axvline(0.80, ls="--", color="#ef4444", lw=1, label="Recall = 0.80")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve – Gradient Boosting", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    logger.info("  Precision-recall curve saved → outputs/precision_recall_curve.png")

    # ── F1 vs threshold curve ─────────────────────────────────────────────────
    thresholds_scan = np.linspace(0.01, 0.99, 200)
    f1_scores = [
        f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds_scan
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds_scan, f1_scores, lw=2, color="#10b981")
    ax.axvline(best_thresh_f1, ls="--", color="#f59e0b", lw=1.5,
               label=f"Best F1 threshold = {best_thresh_f1:.2f}")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("F1 Score",           fontsize=12)
    ax.set_title("F1 Score vs Decision Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "f1_vs_threshold.png", dpi=150)
    plt.close(fig)
    logger.info("  F1 vs threshold curve saved → outputs/f1_vs_threshold.png")

    logger.info("\n✅  Evaluation complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved churn model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "European_Bank.csv"),
        help="Path to the CSV dataset",
    )
    args = parser.parse_args()
    full_evaluation(data_path=args.data)
