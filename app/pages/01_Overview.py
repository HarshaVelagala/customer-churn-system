from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from src.config import PROCESSED_DATA_PATH


def app() -> None:
    st.title("Overview")
    if not PROCESSED_DATA_PATH.exists():
        st.warning("Dataset not found. Run `python main.py` to initialize the system.")
        return
    df = pd.read_csv(PROCESSED_DATA_PATH)
    churn_rate = float(df["churn"].mean())
    st.metric("Customers", len(df))
    st.metric("Churn Rate", f"{churn_rate:.2%}")
    st.metric("Average Credit Limit", f"${df['credit_limit'].mean():,.0f}")
