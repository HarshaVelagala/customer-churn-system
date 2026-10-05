from __future__ import annotations

from typing import Iterable

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["age"] = pd.to_numeric(output.get("age", 0), errors="coerce").fillna(0)
    output["credit_limit"] = pd.to_numeric(output.get("credit_limit", 0), errors="coerce").fillna(0)
    output["months_on_book"] = pd.to_numeric(output.get("months_on_book", 0), errors="coerce").fillna(0)
    output["months_inactive"] = pd.to_numeric(output.get("months_inactive", 0), errors="coerce").fillna(0)
    output["contacts_count"] = pd.to_numeric(output.get("contacts_count", 0), errors="coerce").fillna(0)
    output["total_transaction_count"] = pd.to_numeric(output.get("total_transaction_count", 0), errors="coerce").fillna(0)
    output["total_transaction_amount"] = pd.to_numeric(output.get("total_transaction_amount", 0), errors="coerce").fillna(0)
    output["transaction_count_change"] = pd.to_numeric(output.get("transaction_count_change", 0), errors="coerce").fillna(0)
    output["transaction_amount_change"] = pd.to_numeric(output.get("transaction_amount_change", 0), errors="coerce").fillna(0)
    output["utilization_ratio"] = pd.to_numeric(output.get("utilization_ratio", 0), errors="coerce").fillna(0)
    output["revolving_balance"] = pd.to_numeric(output.get("revolving_balance", 0), errors="coerce").fillna(0)
    output["late_payments"] = pd.to_numeric(output.get("late_payments", 0), errors="coerce").fillna(0)
    output["digital_logins"] = pd.to_numeric(output.get("digital_logins", 0), errors="coerce").fillna(0)
    output["online_transactions"] = pd.to_numeric(output.get("online_transactions", 0), errors="coerce").fillna(0)
    output["customer_service_calls"] = pd.to_numeric(output.get("customer_service_calls", 0), errors="coerce").fillna(0)
    output["dependent_count"] = pd.to_numeric(output.get("dependent_count", 0), errors="coerce").fillna(0)
    if "churn" not in output.columns and "churn_label" in output.columns:
        output["churn"] = output["churn_label"].copy()
    elif "churn" in output.columns and "churn_label" not in output.columns:
        output["churn_label"] = output["churn"].copy()
    return output
