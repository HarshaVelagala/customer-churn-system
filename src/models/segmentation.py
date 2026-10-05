from __future__ import annotations

import pandas as pd
import joblib

from src.config import MODEL_DIR
from src.features.engineering import engineer_features
from src.data.preprocessing import build_preprocessor

FEATURE_COLUMNS = [
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


def load_model_bundle() -> tuple[object, object] | None:
    model_path = MODEL_DIR / "churn_model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    if not model_path.exists() or not preprocessor_path.exists():
        return None
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def _as_feature_frame(record: dict) -> pd.DataFrame:
    row = {column: record.get(column, 0) for column in FEATURE_COLUMNS}
    return pd.DataFrame([row])


def predict_single_record(record: dict) -> dict:
    bundle = load_model_bundle()
    if bundle is None:
        raise FileNotFoundError("Model bundle not found. Run `python main.py` to initialize the system.")
    model, preprocessor = bundle
    frame = engineer_features(_as_feature_frame(record))
    transformed = preprocessor.transform(frame[FEATURE_COLUMNS])
    predicted_probability = float(model.predict_proba(transformed)[0, 1])
    predicted_label = int(model.predict(transformed)[0])
    if predicted_probability >= 0.75:
        risk_level = "Critical"
    elif predicted_probability >= 0.55:
        risk_level = "High"
    elif predicted_probability >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    return {
        "prediction": predicted_label,
        "churn_probability": round(predicted_probability, 4),
        "churn_flag": "Churned" if predicted_label else "Existing",
        "risk_level": risk_level,
    }


def predict_batch(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle()
    if bundle is None:
        raise FileNotFoundError("Model bundle not found. Run `python main.py` to initialize the system.")
    model, preprocessor = bundle
    data = engineer_features(frame.copy())
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for prediction: {missing}")
    transformed = preprocessor.transform(data[FEATURE_COLUMNS])
    probs = model.predict_proba(transformed)[:, 1]
    preds = model.predict(transformed)
    output = data.copy()
    output["churn_probability"] = probs
    output["churn_pred"] = preds
    output["risk_level"] = [
        "Critical" if p >= 0.75 else "High" if p >= 0.55 else "Medium" if p >= 0.35 else "Low" for p in probs
    ]
    return output
