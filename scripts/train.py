from __future__ import annotations

from src.features.engineering import engineer_features
from src.models.churn_model import train_model
from src.models.segmentation import train_segmentation_model
from src.data.loader import load_customer_data


if __name__ == "__main__":
    data = load_customer_data()
    train_model()
    train_segmentation_model(data)
    print("Training pipeline complete.")
