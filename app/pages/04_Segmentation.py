from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

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
    counts = segmented.groupby("segment").size().reset_index(name="customers")
    st.dataframe(counts, use_container_width=True)
