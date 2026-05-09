"""
predict.py
----------
Batch inference script.
Loads the saved model and scores a new CSV file (without the Exited column).

Usage
-----
python src/predict.py --input new_customers.csv --output predictions.csv

The output CSV will contain the original columns plus:
  churn_probability   – raw score [0, 1]
  risk_band_default   – Low / Medium / High (fixed thresholds)
  risk_band_optimized – Low / Medium / High (uses best F1 threshold)
  churn_flag          – 1 if prob >= best_threshold_f1 else 0
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    get_logger,
    assign_risk_band,
    assign_risk_band_optimized,
    PROJECT_ROOT,
    ID_COLS,
    TARGET,
)

warnings.filterwarnings("ignore")
logger = get_logger("predict")


def load_model_and_threshold():
    """Load the saved pipeline and the best decision threshold."""
    models_dir     = PROJECT_ROOT / "models"
    model_path     = models_dir / "model.pkl"
    threshold_path = models_dir / "best_threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )

    pipeline = joblib.load(model_path)

    with open(threshold_path) as f:
        cfg = json.load(f)
    best_thresh = cfg["best_threshold_f1"]
    logger.info(f"Loaded model from {model_path}  (best threshold: {best_thresh:.4f})")
    return pipeline, best_thresh


def predict_batch(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Score a batch of customers from a CSV file.

    Parameters
    ----------
    input_csv  : Path to input CSV (same schema as training data, Exited optional)
    output_csv : Path to save predictions CSV (optional)

    Returns
    -------
    DataFrame with original columns + scoring columns appended
    """
    pipeline, best_thresh = load_model_and_threshold()

    df_raw = pd.read_csv(input_csv)
    logger.info(f"Input shape: {df_raw.shape}")

    # Drop target if present (for easy re-scoring of training data)
    df = df_raw.drop(columns=[TARGET], errors="ignore")

    # Drop ID / admin columns – not used by the model
    cols_to_drop = [c for c in (ID_COLS + ["Year"]) if c in df.columns]
    df_model = df.drop(columns=cols_to_drop)

    # Predict
    y_prob = pipeline.predict_proba(df_model)[:, 1]
    y_flag = (y_prob >= best_thresh).astype(int)

    # Attach scoring columns to the original dataframe
    df_raw["churn_probability"]   = np.round(y_prob, 4)
    df_raw["risk_band_default"]   = assign_risk_band(y_prob)
    df_raw["risk_band_optimized"] = assign_risk_band_optimized(y_prob, best_thresh)
    df_raw["churn_flag"]          = y_flag

    if output_csv:
        df_raw.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved → {output_csv}")

    # Summary statistics
    logger.info("\n── Prediction Summary ──")
    logger.info(f"  Total customers scored : {len(df_raw)}")
    logger.info(f"  Flagged as churn       : {y_flag.sum()} ({y_flag.mean()*100:.1f}%)")
    logger.info(
        "\nRisk band distribution:\n"
        + df_raw["risk_band_default"].value_counts().to_string()
    )

    return df_raw


def predict_single(customer: dict) -> dict:
    """
    Score a single customer provided as a dictionary.
    Used by the Streamlit app.

    Parameters
    ----------
    customer : dict with keys matching feature columns (without Exited)

    Returns
    -------
    dict with keys: churn_probability, risk_band, churn_flag, threshold_used
    """
    pipeline, best_thresh = load_model_and_threshold()

    df = pd.DataFrame([customer])

    # Drop ID / admin columns if accidentally included
    cols_to_drop = [c for c in (ID_COLS + ["Year", TARGET]) if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    prob  = float(pipeline.predict_proba(df)[0, 1])
    band  = str(assign_risk_band(np.array([prob]))[0])
    flag  = int(prob >= best_thresh)

    return {
        "churn_probability": round(prob, 4),
        "risk_band":         band,
        "churn_flag":        flag,
        "threshold_used":    round(best_thresh, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch churn scoring")
    parser.add_argument("--input",  type=str, required=True, help="Input CSV path")
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "outputs" / "predictions.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    predict_batch(input_csv=args.input, output_csv=args.output)
