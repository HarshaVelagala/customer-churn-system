from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DATA_PATH


def app() -> None:
    st.title("Overview")
    if not PROCESSED_DATA_PATH.exists():
        st.warning("Dataset not found. Run `python main.py` to initialize the system.")
        return
    df = pd.read_csv(PROCESSED_DATA_PATH)
    churn_rate = df["churn"].mean() if "churn" in df.columns else 0.0
    st.metric("Customers", len(df))
    st.metric("Churn rate", f"{churn_rate:.2%}")
    st.metric("Average credit limit", f"${df['credit_limit'].mean():,.0f}")
