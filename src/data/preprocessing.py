from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_DIR

NUMERIC_FEATURES = [
    "age",
    "credit_limit",
    "months_on_book",
    "months_inactive",
    "total_relationship_count",
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

CATEGORICAL_FEATURES = [
    "gender",
    "income_bracket",
    "card_type",
    "education_level",
    "marital_status",
]

TARGET_COLUMN = "churn"


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, metadata_path: str | Path | None = None) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    if metadata_path is not None:
        meta = Path(metadata_path)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}, indent=2), encoding="utf-8")


def load_preprocessor() -> ColumnTransformer | None:
    model_path = MODEL_DIR / "preprocessor.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)
