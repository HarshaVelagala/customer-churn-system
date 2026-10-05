from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans

from src.config import MODEL_DIR

SEGMENT_LABELS = {
    0: "Loyal",
    1: "High Value",
    2: "Low Engagement",
    3: "At Risk",
}


def train_segmentation_model(df: pd.DataFrame, n_clusters: int = 4, random_seed: int = 42) -> KMeans:
    numeric = [
        "age",
        "credit_limit",
        "months_on_book",
        "months_inactive",
        "contacts_count",
        "total_transaction_count",
        "total_transaction_amount",
        "utilization_ratio",
        "late_payments",
        "digital_logins",
        "customer_service_calls",
    ]
    features = df[numeric].fillna(0)
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    model.fit(features)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "segmentation_model.joblib")
    return model


def load_segmentation_model() -> KMeans | None:
    path = MODEL_DIR / "segmentation_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def assign_segments(df: pd.DataFrame, model: KMeans | None = None) -> pd.DataFrame:
    if model is None:
        model = load_segmentation_model()
    if model is None:
        raise FileNotFoundError("Segmentation model not found. Run `python main.py` to initialize the system.")

    numeric = [
        "age",
        "credit_limit",
        "months_on_book",
        "months_inactive",
        "contacts_count",
        "total_transaction_count",
        "total_transaction_amount",
        "utilization_ratio",
        "late_payments",
        "digital_logins",
        "customer_service_calls",
    ]
    output = df.copy()
    output["segment_id"] = model.predict(output[numeric].fillna(0))
    output["segment"] = output["segment_id"].map(SEGMENT_LABELS)
    return output
