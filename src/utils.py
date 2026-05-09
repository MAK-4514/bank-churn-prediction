"""
utils.py
--------
Shared utility functions for the Bank Churn Prediction project.
Handles logging, directory setup, and helper methods used by
train.py, evaluate.py, and predict.py.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────

def get_logger(name: str = "churn_model") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ──────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # d:/project/


def ensure_dirs() -> None:
    """Create required output directories if they don't already exist."""
    for d in ["models", "outputs", "data"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

# Columns to drop before modelling
ID_COLS = ["CustomerId", "Surname"]

# Target variable
TARGET = "Exited"

# Numerical feature columns
NUM_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

# Categorical feature columns (will be one-hot encoded)
CAT_FEATURES = ["Geography", "Gender"]


def load_data(
    filepath: str,
    drop_year: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load raw CSV and perform initial column hygiene.

    Parameters
    ----------
    filepath    : Path to European_Bank.csv (or similar)
    drop_year   : Whether to drop the 'Year' column.
                  Year is dropped by default because it acts as a
                  record-entry date unrelated to customer behaviour;
                  keeping it can introduce spurious temporal leakage.
    random_state: Unused here but kept for API consistency.

    Returns
    -------
    Cleaned DataFrame ready for the preprocessing pipeline.
    """
    df = pd.read_csv(filepath)

    # Drop administrative / non-predictive identifiers
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    if drop_year and "Year" in df.columns:
        cols_to_drop.append("Year")
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    return df


# ──────────────────────────────────────────────
# Risk-banding helpers
# ──────────────────────────────────────────────

# Default fixed thresholds (business-defined)
DEFAULT_LOW_THRESHOLD = 0.40
DEFAULT_HIGH_THRESHOLD = 0.70


def assign_risk_band(
    probabilities: np.ndarray,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> np.ndarray:
    """
    Convert raw churn probabilities into categorical risk bands.

    Bands
    -----
    Low    : p < low_threshold
    Medium : low_threshold <= p < high_threshold
    High   : p >= high_threshold
    """
    bands = np.where(
        probabilities >= high_threshold,
        "High",
        np.where(probabilities >= low_threshold, "Medium", "Low"),
    )
    return bands


def assign_risk_band_optimized(
    probabilities: np.ndarray,
    opt_threshold: float,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> np.ndarray:
    """
    Alternative banding that uses the *optimized* decision threshold
    (found during training) as the Low/Medium boundary.
    This makes the 'churn flag' consistent with the risk band.

    Bands
    -----
    Low    : p < opt_threshold
    Medium : opt_threshold <= p < high_threshold
    High   : p >= high_threshold
    """
    bands = np.where(
        probabilities >= high_threshold,
        "High",
        np.where(probabilities >= opt_threshold, "Medium", "Low"),
    )
    return bands


# ──────────────────────────────────────────────
# Metrics / artefact I/O
# ──────────────────────────────────────────────

def save_metrics(metrics: dict, path: str) -> None:
    """Persist a metrics dictionary to JSON, converting numpy types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, default=_convert)


def load_metrics(path: str) -> dict:
    """Load a metrics JSON file."""
    with open(path) as f:
        return json.load(f)
