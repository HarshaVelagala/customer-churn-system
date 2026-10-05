from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DATA_PATH


def app() -> None:
    st.title("Data Explorer")
    if not PROCESSED_DATA_PATH.exists():
        st.warning("Dataset not found. Run `python main.py` to initialize the system.")
        return
    df = pd.read_csv(PROCESSED_DATA_PATH)
    st.dataframe(df.head(200), use_container_width=True)
