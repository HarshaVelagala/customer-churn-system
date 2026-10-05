from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

PAGES = {
    "Overview": "app.pages.01_Overview",
    "Data Explorer": "app.pages.02_Data_Explorer",
    "Churn Predictor": "app.pages.03_Churn_Predictor",
    "Segmentation": "app.pages.04_Segmentation",
    "Business Analytics": "app.pages.05_Business_Analytics",
    "AI Retention Advisor": "app.pages.06_Retention_Advisor",
}


st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")
st.sidebar.title("Customer Churn System")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
module = __import__(PAGES[selection], fromlist=["*"])
module.app()
