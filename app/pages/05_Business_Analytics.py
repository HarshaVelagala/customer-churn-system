from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from src.analytics.sql_analytics import ensure_database, query_database
from src.config import DATABASE_PATH, PROCESSED_DATA_PATH


def app() -> None:
    st.title("Business Analytics")
    if not PROCESSED_DATA_PATH.exists():
        st.warning("Dataset not found. Run `python main.py` to initialize the system.")
        return
    ensure_database(PROCESSED_DATA_PATH, DATABASE_PATH)
    metrics = query_database(DATABASE_PATH, "SELECT COUNT(*) AS total_customers, SUM(churn_label) AS churned_customers, ROUND(AVG(total_transaction_amount), 2) AS avg_transaction_amount, ROUND(100.0*SUM(churn_label)/COUNT(*), 2) AS churn_rate FROM customers")
    st.dataframe(metrics, use_container_width=True)
