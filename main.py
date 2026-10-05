from __future__ import annotations

from src.config import BASE_DIR
from src.data.generator import generate_customer_data
from src.features.engineering import train_model
from src.models.segmentation import train_segmentation_model
from src.analytics.sql_analytics import ensure_database
from src.data.loader import load_customer_data


def initialize() -> dict:
    from src.config import DATABASE_PATH, PROCESSED_DATA_PATH
    import pandas as pd

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if not PROCESSED_DATA_PATH.exists():
        generate_customer_data(5000, seed=42, output_path=PROCESSED_DATA_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    train_model(random_seed=42)
    train_segmentation_model(df)
    ensure_database(PROCESSED_DATA_PATH, DATABASE_PATH)

    print("SYSTEM INITIALIZATION COMPLETE")
    print(f"Dataset: {len(df)} customers")
    print("Model: XGBoost")
    with open(BASE_DIR / "models" / "metrics.json", "r", encoding="utf-8") as handle:
        metrics = __import__("json").load(handle)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Segments: 4")
    print("Database: Ready")
    return {"status": "ok", "dataset_size": len(df), "model": "XGBoost"}


if __name__ == "__main__":
    initialize()
