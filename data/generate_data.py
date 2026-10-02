"""Generate a realistic synthetic telecom customer churn dataset.

The churn labels are produced from a logistic model over the features, so the
dataset has learnable, plausible structure: short tenure, month-to-month
contracts, high monthly charges, fibre-optic internet without tech support,
electronic-cheque billing and many support calls all push churn probability up.

Usage:
    python data/generate_data.py [--rows 5000] [--seed 42] [--out data/churn.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:05d}" for i in range(1, n_rows + 1)]
    gender = rng.choice(["Male", "Female"], n_rows)
    senior_citizen = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])

    # Tenure in months: mixture of new and long-standing customers.
    tenure = np.clip(
        rng.choice(
            [
                rng.integers(0, 13, n_rows),   # new customers
                rng.integers(12, 73, n_rows),  # established customers
            ]
        ),
        0,
        72,
    ).astype(int)

    phone_service = rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n_rows, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22]
    )
    has_internet = internet_service != "No"

    def _addon(p_yes: float) -> np.ndarray:
        choice = rng.choice(["Yes", "No"], n_rows, p=[p_yes, 1 - p_yes])
        return np.where(has_internet, choice, "No internet service")

    online_security = _addon(0.30)
    online_backup = _addon(0.34)
    device_protection = _addon(0.34)
    tech_support = _addon(0.29)
    streaming_tv = _addon(0.38)
    streaming_movies = _addon(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_rows,
        p=[0.34, 0.19, 0.22, 0.25],
    )

    # Monthly charges depend on services subscribed.
    base = 20.0
    monthly_charges = (
        base
        + np.where(internet_service == "Fiber optic", 35.0, 0.0)
        + np.where(internet_service == "DSL", 20.0, 0.0)
        + np.where(phone_service == "Yes", 10.0, 0.0)
        + np.where(streaming_tv == "Yes", 8.0, 0.0)
        + np.where(streaming_movies == "Yes", 8.0, 0.0)
        + np.where(device_protection == "Yes", 5.0, 0.0)
        + np.where(online_backup == "Yes", 4.0, 0.0)
        + rng.normal(0, 4, n_rows)
    )
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0).round(2)

    total_charges = np.round(
        np.clip(tenure * monthly_charges * rng.uniform(0.9, 1.1, n_rows), 0, None), 2
    )

    # Support calls in the last 6 months: skewed towards zero.
    support_calls = np.clip(
        rng.negative_binomial(n=2, p=0.55, size=n_rows), 0, 10
    ).astype(int)

    # ---- Churn label via a logistic model over the features ----
    logit = (
        -2.2
        + 2.20 * (contract == "Month-to-month")
        + 0.80 * (contract == "One year")
        + 1.20 * (internet_service == "Fiber optic")
        + 1.00 * (tech_support == "No")
        + 0.70 * (online_security == "No")
        + 0.80 * (payment_method == "Electronic check")
        + 0.50 * (paperless_billing == "Yes")
        + 0.45 * (senior_citizen == 1)
        + 0.40 * (streaming_movies == "Yes")
        - 0.90 * np.log1p(tenure)                       # loyal customers stay
        + 0.020 * (monthly_charges - 65.0)               # expensive plans churn
        + 0.45 * support_calls                           # bad experiences churn
        - 0.40 * (partner == "Yes")
        + rng.normal(0, 0.25, n_rows)                    # unobserved noise
    )
    churn_prob = _sigmoid(logit)
    churn = (rng.random(n_rows) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "churn": churn,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/churn.csv")
    args = parser.parse_args()

    df = generate(n_rows=args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out} "
          f"(churn rate: {df['churn'].mean():.1%})")


if __name__ == "__main__":
    main()
