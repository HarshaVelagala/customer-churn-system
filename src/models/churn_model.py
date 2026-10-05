from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import BASE_DIR, MODEL_DIR, RANDOM_SEED
from src.data.generator import generate_customer_data
from src.data.loader import load_customer_data
from src.data.preprocessing import build_preprocessor, save_preprocessor
from src.features.engineering import engineer_features

TARGET_COLUMN = "churn"


def train_model(random_seed: int = RANDOM_SEED) -> tuple[XGBClassifier, object, dict]:
    data = load_customer_data()
    df = engineer_features(data)
    feature_columns = [
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

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    preprocessor = build_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train_processed, y_train)

    test_proba = model.predict_proba(X_test_processed)[:, 1]
    test_pred = model.predict(X_test_processed)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "churn_model.joblib")
    save_preprocessor(preprocessor)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return model, preprocessor, metrics


if __name__ == "__main__":
    model, preprocessor, metrics = train_model()
    print(f"Model trained. ROC-AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
