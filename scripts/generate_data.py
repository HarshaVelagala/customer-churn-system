from __future__ import annotations

from src.ai.retention_advisor import generate_retention_strategy
from src.config import ensure_directories
from src.data.generator import generate_customer_data
from src.features.engineering import engineer_features
from src.models.churn_model import train_model
from src.models.segmentation import train_segmentation_model
from src.analytics.sql_analytics import ensure_database


def initialize() -> dict:
    ensure_directories()
    from src.config import PROCESSED_DATA_PATH, DATABASE_PATH

    try:
        import pandas as pd
        csv_exists = PROCESSED_DATA_PATH.exists()
        if not csv_exists:
            generate_customer_data(5000, seed=42, output_path=PROCESSED_DATA_PATH)
        df = pd.read_csv(PROCESSED_DATA_PATH)
        train_model(random_seed=42)
        train_segmentation_model(df)
        ensure_database(PROCESSED_DATA_PATH, DATABASE_PATH)
        return {"status": "ok", "rows": len(df)}
    except Exception as exc:
        raise RuntimeError(f"Initialization failed: {exc}") from exc


if __name__ == "__main__":
    result = initialize()
    print(result)
