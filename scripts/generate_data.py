from __future__ import annotations

import json

import pandas as pd

from src.analytics.sql_analytics import ensure_database
from src.config import DATABASE_PATH, PROCESSED_DATA_PATH, ensure_directories
from src.data.generator import generate_customer_data
from src.features.engineering import train_model
from src.models.segmentation import train_segmentation_model


def initialize() -> dict:
    ensure_directories()
    if not PROCESSED_DATA_PATH.exists():
        generate_customer_data(5000, seed=42, output_path=PROCESSED_DATA_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    _, _, metrics = train_model(random_seed=42)
    train_segmentation_model(df, n_clusters=4, random_seed=42)
    ensure_database(PROCESSED_DATA_PATH, DATABASE_PATH)

    print("SYSTEM INITIALIZATION COMPLETE")
    print(f"Dataset: {len(df)} customers")
    print("Model: XGBoost")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Segments: 4")
    print("Database: Ready")
    return {"status": "ok", "dataset_size": len(df), "roc_auc": metrics["roc_auc"], "accuracy": metrics["accuracy"]}


if __name__ == "__main__":
    initialize()
