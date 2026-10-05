from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DATA_PATH
from src.models.segmentation import assign_segments, load_segmentation_model


def app() -> None:
    st.title("Customer Segmentation")
    if not PROCESSED_DATA_PATH.exists():
        st.warning("Dataset not found. Run `python main.py` to initialize the system.")
        return
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = load_segmentation_model()
    if model is None:
        st.warning("Segmentation model missing. Run `python main.py` to initialize the system.")
        return
    segmented = assign_segments(df, model)
    summary = segmented.groupby("segment").agg(customers=("customer_id", "count"), churn_rate=("churn", "mean")).reset_index()
    st.dataframe(summary, use_container_width=True)
