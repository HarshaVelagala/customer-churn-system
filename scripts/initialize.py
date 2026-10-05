from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.sql_analytics import ensure_database, query_database
from src.ai.retention_advisor import generate_retention_strategy
from src.config import BASE_DIR, DATABASE_PATH, PROCESSED_DATA_PATH, MODEL_DIR
from src.data.loader import load_customer_data
from src.models.inference import assign_segments, load_segmentation_model


def initialize_system(n_customers: int = 5000, random_seed: int = 42) -> dict:
    from src.data.generator import generate_customer_data
    from src.models.churn_model import predict_batch
    from src.models.segmentation import train_segmentation_model
    from src.features.engineering import engineer_features
    from src.models.churn_model import train_model

    data_path = PROCESSED_DATA_PATH
    if not data_path.exists():
        df = generate_customer_data(n_customers=n_customers, seed=random_seed, output_path=data_path)
    else:
        df = pd.read_csv(data_path)

    model, preprocessor, metrics = train_model(random_seed=random_seed)

    segmentation_model = train_segmentation_model(df)
    ensure_database(data_path, DATABASE_PATH)

    analytics = query_database(DATABASE_PATH, "SELECT COUNT(*) AS total_customers, SUM(churn_label) AS churned_customers FROM customers")
    summary = {
        "dataset_size": len(df),
        "model_type": "XGBoost",
        "roc_auc": metrics.get("roc_auc"),
        "accuracy": metrics.get("accuracy"),
        "segments": 4,
        "database_ready": True,
        "analytics": analytics.to_dict(orient="records")[0],
    }
    print("SYSTEM INITIALIZATION COMPLETE")
    print(f"Dataset: {summary['dataset_size']} customers")
    print(f"Model: {summary['model_type']}")
    print(f"ROC-AUC: {summary['roc_auc']:.4f}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print("Segments: 4")
    print("Database: Ready")
    return summary


if __name__ == "__main__":
    initialize_system()
