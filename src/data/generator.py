from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_PATH, RANDOM_SEED


def generate_customer_data(n_customers: int = 5000, seed: int | None = None, output_path: str | Path | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
    n = int(n_customers)

    customer_ids = [f"CUST_{index:06d}" for index in range(1, n + 1)]
    age = rng.normal(42, 12, n).clip(21, 75).astype(int)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    income_bracket = rng.choice(["<$40K", "$40K-$60K", "$60K-$80K", "$80K-$120K", "$120K+"], size=n, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    education_level = rng.choice(["High School", "Some College", "Graduate", "Post-Graduate"], size=n, p=[0.18, 0.22, 0.38, 0.22])
    marital_status = rng.choice(["Married", "Single", "Divorced"], size=n, p=[0.55, 0.35, 0.10])
    dependent_count = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.20, 0.18, 0.25, 0.22, 0.10, 0.05])
    card_type = rng.choice(["Blue", "Silver", "Gold", "Platinum"], size=n, p=[0.53, 0.25, 0.15, 0.07])

    months_on_book = rng.integers(6, 120, size=n)
    credit_limit = np.clip(np.random.lognormal(mean=9.3, sigma=0.7, size=n) * 100, 1000, 35000).round(2)
    revolving_balance = np.clip(credit_limit * rng.beta(1.5, 4.0, size=n), 0, credit_limit).round(2)
    utilization_ratio = np.clip(revolving_balance / credit_limit, 0.0, 1.0).round(4)
    avg_utilization = utilization_ratio.copy()

    total_transaction_amount = np.clip(np.random.lognormal(mean=8.4, sigma=0.9, size=n) * 120, 300, 40000).round(2)
    total_transaction_count = rng.normal(72, 26, n).clip(8, 220).round().astype(int)
    transaction_count_change = rng.normal(0.8, 0.3, n).clip(-0.5, 2.5).round(3)
    transaction_amount_change = rng.normal(0.9, 0.35, n).clip(-0.5, 3.0).round(3)

    months_inactive = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n, p=[0.25, 0.22, 0.20, 0.14, 0.10, 0.06, 0.03])
    contacts_count = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n, p=[0.42, 0.22, 0.16, 0.10, 0.06, 0.03, 0.01])
    late_payments = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.52, 0.20, 0.13, 0.08, 0.05, 0.02])
    digital_logins = rng.integers(0, 35, size=n)
    online_transactions = rng.integers(0, 25, size=n)
    customer_service_calls = rng.integers(0, 8, size=n)
    total_relationship_count = rng.integers(1, 8, size=n)

    churn_score = (
        0.22 * (months_inactive / 6.0)
        + 0.20 * np.clip((1.0 - transaction_count_change / 2.5), 0.0, 1.0)
        + 0.16 * (contacts_count / 6.0)
        + 0.12 * (late_payments / 5.0)
        + 0.14 * np.clip((1.0 - digital_logins / 35.0), 0.0, 1.0)
        + 0.10 * np.clip((1.0 - utilization_ratio), 0.0, 1.0)
        + 0.06 * rng.uniform(0.0, 1.0, n)
    )
    churn = (churn_score > 0.52).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": age,
            "gender": gender,
            "income": income_bracket,
            "income_bracket": income_bracket,
            "credit_limit": credit_limit,
            "months_on_book": months_on_book,
            "months_inactive": months_inactive,
            "months_inactive_12m": months_inactive,
            "total_relationship_count": total_relationship_count,
            "contacts_count": contacts_count,
            "contacts_count_12m": contacts_count,
            "total_transaction_count": total_transaction_count,
            "total_transaction_amount": total_transaction_amount,
            "transaction_count_change": transaction_count_change,
            "transaction_amount_change": transaction_amount_change,
            "utilization_ratio": utilization_ratio,
            "avg_utilization": avg_utilization,
            "revolving_balance": revolving_balance,
            "late_payments": late_payments,
            "digital_logins": digital_logins,
            "digital_logins_30d": digital_logins,
            "online_transactions": online_transactions,
            "customer_service_calls": customer_service_calls,
            "card_category": card_type,
            "card_type": card_type,
            "education": education_level,
            "education_level": education_level,
            "marital_status": marital_status,
            "dependent_count": dependent_count,
            "churn": churn,
            "churn_label": churn,
        }
    )

    output = Path(output_path) if output_path is not None else PROCESSED_DATA_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


if __name__ == "__main__":
    generate_customer_data(5000, seed=RANDOM_SEED)
    print(f"Dataset generated: {PROCESSED_DATA_PATH}")
