"""
generate_synthetic_data.py
--------------------------
Generates a synthetic dataset that closely mirrors the European Bank
Customer Churn dataset schema and statistical profile (~10,000 rows).

Run
---
python src/generate_synthetic_data.py

Output: data/European_Bank.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_ROWS = 10_000

rng = np.random.default_rng(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / "European_Bank.csv"


def generate() -> pd.DataFrame:
    # ── IDs and admin ────────────────────────────────────────────────────────
    customer_ids = rng.integers(15_000_000, 16_000_000, size=N_ROWS)
    surnames = rng.choice(
        ["Smith", "Jones", "Müller", "Dupont", "Garcia",
         "Brown", "Taylor", "Wilson", "Martin", "Hoffman"],
        size=N_ROWS,
    )
    years = rng.choice([2016, 2017, 2018], size=N_ROWS)

    # ── Geography & Gender ───────────────────────────────────────────────────
    geography = rng.choice(["France", "Spain", "Germany"],
                           p=[0.50, 0.25, 0.25], size=N_ROWS)
    gender = rng.choice(["Male", "Female"], p=[0.54, 0.46], size=N_ROWS)

    # ── Continuous features ──────────────────────────────────────────────────
    credit_score = np.clip(
        rng.normal(650, 97, N_ROWS).astype(int), 350, 850
    )
    age = np.clip(rng.normal(38.9, 10.5, N_ROWS).astype(int), 18, 92)
    tenure = rng.integers(0, 11, size=N_ROWS)

    # ~30% of customers have zero balance (typical in churn datasets)
    has_balance = rng.random(N_ROWS) > 0.29
    balance = np.where(
        has_balance,
        np.clip(rng.normal(76_485, 62_397, N_ROWS), 0, 250_898),
        0.0,
    ).round(2)

    num_products = rng.choice([1, 2, 3, 4],
                              p=[0.50, 0.46, 0.026, 0.014], size=N_ROWS)
    has_cr_card   = rng.choice([0, 1], p=[0.29, 0.71], size=N_ROWS)
    is_active     = rng.choice([0, 1], p=[0.49, 0.51], size=N_ROWS)
    salary = np.clip(
        rng.uniform(11.58, 199_992, N_ROWS), 0, 200_000
    ).round(2)

    # ── Synthetic target (Exited) ────────────────────────────────────────────
    # Logistic-regression-style log-odds to reproduce ~20% churn rate
    # with realistic feature correlations
    log_odds = (
        -0.5
        + 0.03  * (age - 38)                               # older → more churn
        + 0.50  * (geography == "Germany").astype(float)   # Germany premium
        + 0.30  * (gender == "Female").astype(float)       # slight female effect
        - 0.80  * is_active                                # active → retained
        - 0.60  * (num_products == 2).astype(float)        # 2 products → retained
        + 1.20  * (num_products >= 3).astype(float)        # 3-4 products → churn (overfitting)
        + 0.40  * (balance > 100_000).astype(float)        # high balance paradox
        - 0.003 * (credit_score - 500)                     # higher score → slight retention
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    exited = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "Year":            years,
        "CustomerId":      customer_ids,
        "Surname":         surnames,
        "CreditScore":     credit_score,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCrCard":       has_cr_card,
        "IsActiveMember":  is_active,
        "EstimatedSalary": salary,
        "Exited":          exited,
    })

    actual_churn_rate = exited.mean() * 100
    print(f"Generated {N_ROWS:,} rows  |  Churn rate: {actual_churn_rate:.1f}%")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    generate()
