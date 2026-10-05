from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.retention_advisor import generate_retention_strategy
from src.analytics.sql_analytics import ensure_database
from src.config import DATABASE_PATH, PROCESSED_DATA_PATH, ensure_directories
from src.data.generator import generate_customer_data
from src.features.engineering import train_model
from src.models.segmentation import train_segmentation_model


def initialize() -> dict:
    ensure_directories()
    if not PROCESSED_DATA_PATH.exists():
        generate_customer_data(5000, seed=42, output_path=PROCESSED_DATA_PATH)

    data = __import__("pandas").read_csv(PROCESSED_DATA_PATH)
    _, _, metrics = train_model(random_seed=42)
    train_segmentation_model(data, n_clusters=4, random_seed=42)
    ensure_database(PROCESSED_DATA_PATH, DATABASE_PATH)

    print("SYSTEM INITIALIZATION COMPLETE")
    print(f"Dataset: {len(data)} customers")
    print("Model: XGBoost")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Segments: 4")
    print("Database: Ready")
    return {"status": "ok", "dataset_size": len(data), "model": "XGBoost"}


if __name__ == "__main__":
    initialize()
