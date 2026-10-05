from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import BASE_DIR, MODEL_DIR
from src.data.loader import load_customer_data
from src.data.preprocessing import build_preprocessor, load_preprocessor
from src.features.engineering import engineer_features


def load_model_bundle() -> tuple[object, object] | None:
    model_path = MODEL_DIR / "churn_model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    if not model_path.exists() or not preprocessor_path.exists():
        return None
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def _single_record_to_dataframe(record: dict) -> pd.DataFrame:
    required = [
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
    row = {key: record.get(key, 0) for key in required}
    return pd.DataFrame([row])


def predict_single_record(record: dict) -> dict:
    bundle = load_model_bundle()
    if bundle is None:
        raise FileNotFoundError("Model bundle not found. Run `python main.py` to initialize the system.")

    model, preprocessor = bundle
    features = _single_record_to_dataframe(record)
    features = engineer_features(features)
    transformed = preprocessor.transform(features)
    probability = float(model.predict_proba(transformed)[0, 1])
    prediction = int(model.predict(transformed)[0])

    if probability >= 0.75:
        risk_level = "Critical"
    elif probability >= 0.55:
        risk_level = "High"
    elif probability >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "churn_label": prediction,
        "risk_level": risk_level,
        "churn_flag": "Churned" if prediction == 1 else "Existing",
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle()
    if bundle is None:
        raise FileNotFoundError("Model bundle not found. Run `python main.py` to initialize the system.")

    model, preprocessor = bundle
    data = engineer_features(df.copy())
    required_columns = [
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
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for prediction: {missing}")

    transformed = preprocessor.transform(data[required_columns])
    probability = model.predict_proba(transformed)[:, 1]
    prediction = model.predict(transformed)
    output = data.copy()
    output["churn_probability"] = probability
    output["churn_pred"] = prediction
    output["risk_level"] = [
        "Critical" if p >= 0.75 else "High" if p >= 0.55 else "Medium" if p >= 0.35 else "Low"
        for p in probability
    ]
    return output
