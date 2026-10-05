from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ai.retention_advisor import generate_retention_strategy
from src.data.generator import generate_customer_data
from src.data.loader import load_customer_data
from src.data.preprocessing import build_preprocessor
from src.features.engineering import engineer_features
from src.models.churn_model import predict_batch, predict_single_record, load_model_bundle
from src.models.segmentation import train_segmentation_model, assign_segments
from src.analytics.sql_analytics import ensure_database, query_database


def test_generate_data():
    df = generate_customer_data(120, seed=7)
    assert len(df) == 120
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_preprocessing():
    df = generate_customer_data(150, seed=9)
    processor = build_preprocessor()
    X = df[["age", "gender", "income_bracket", "card_type", "education_level", "marital_status", "credit_limit", "months_on_book", "months_inactive", "contacts_count", "total_transaction_count", "total_transaction_amount", "transaction_count_change", "transaction_amount_change", "utilization_ratio", "revolving_balance", "late_payments", "digital_logins", "online_transactions", "customer_service_calls", "dependent_count"]]
    transformed = processor.fit_transform(X)
    assert transformed.shape[0] == len(df)


def test_model_training():
    from src.features.engineering import train_model

    _, _, metrics = train_model(random_seed=13)
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_single_prediction():
    record = {
        "age": 38,
        "gender": "Male",
        "income_bracket": "$40K-$60K",
        "card_type": "Blue",
        "education_level": "Graduate",
        "marital_status": "Married",
        "credit_limit": 9000,
        "months_on_book": 48,
        "months_inactive": 2,
        "contacts_count": 1,
        "total_transaction_count": 60,
        "total_transaction_amount": 4000,
        "transaction_count_change": 0.7,
        "transaction_amount_change": 0.8,
        "utilization_ratio": 0.2,
        "revolving_balance": 1800,
        "late_payments": 0,
        "digital_logins": 12,
        "online_transactions": 10,
        "customer_service_calls": 1,
        "dependent_count": 1,
    }
    result = predict_single_record(record)
    assert "churn_probability" in result
    assert result["risk_level"] in {"Low", "Medium", "High", "Critical"}


def test_batch_prediction():
    df = generate_customer_data(80, seed=11)
    predictions = predict_batch(df.head(20))
    assert "churn_probability" in predictions.columns
    assert "risk_level" in predictions.columns


def test_segmentation():
    df = generate_customer_data(100, seed=12)
    model = train_segmentation_model(df, n_clusters=4)
    result = assign_segments(df, model)
    assert "segment" in result.columns
    assert set(result["segment"].unique()) <= {"Loyal", "High Value", "Low Engagement", "At Risk"}


def test_database():
    df = generate_customer_data(40, seed=17)
    db_path = Path("database/test_churn.db")
    ensure_database(df, db_path)
    rows = query_database(db_path, "SELECT COUNT(*) AS n FROM customers")
    assert rows["n"].iloc[0] == 40


def test_retention_engine():
    customer = {"months_inactive": 4, "contacts_count": 2, "late_payments": 2, "digital_logins": 8}
    prediction = {"risk_level": "High", "churn_probability": 0.72}
    result = generate_retention_strategy(customer, prediction)
    assert "strategy_title" in result
    assert "priority" in result or "risk_explanation" in result


def test_generate_data_2():
    df = generate_customer_data(30, seed=18)
    assert df["churn"].sum() >= 0


if __name__ == "__main__":
    test_generate_data()
    test_preprocessing()
    test_model_training()
    test_single_prediction()
    test_batch_prediction()
    test_segmentation()
    test_database()
    test_retention_engine()
    print("All tests passed.")
