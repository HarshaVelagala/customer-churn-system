from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = [
    "age",
    "gender",
    "income_bracket",
    "card_type",
    "education_level",
    "marital_status",
    "credit_limit",
    "months_on_book",
    "months_inactive",
    "contacts_count",
    "total_transaction_count",
    "total_transaction_amount",
    "transaction_count_change",
    "transaction_amount_change",
    "utilization_ratio",
    "revolving_balance",
    "late_payments",
    "digital_logins",
    "online_transactions",
    "customer_service_calls",
    "dependent_count",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["credit_limit"] = pd.to_numeric(df.get("credit_limit", 0), errors="coerce").fillna(0)
    df["utilization_ratio"] = pd.to_numeric(df.get("utilization_ratio", 0), errors="coerce").fillna(0)
    df["months_inactive"] = pd.to_numeric(df.get("months_inactive", 0), errors="coerce").fillna(0)
    df["late_payments"] = pd.to_numeric(df.get("late_payments", 0), errors="coerce").fillna(0)
    df["digital_logins"] = pd.to_numeric(df.get("digital_logins", 0), errors="coerce").fillna(0)
    df["revolving_balance"] = pd.to_numeric(df.get("revolving_balance", 0), errors="coerce").fillna(0)
    df["churn"] = pd.to_numeric(df.get("churn", df.get("churn_label", 0)), errors="coerce").fillna(0).astype(int)
    df["churn_label"] = df["churn"].copy()
    return df
