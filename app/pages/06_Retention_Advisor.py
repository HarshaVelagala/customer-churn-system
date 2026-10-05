from __future__ import annotations

import streamlit as st

from src.ai.retention_advisor import generate_retention_strategy


def app() -> None:
    st.title("AI Retention Advisor")
    customer = {
        "months_inactive": st.number_input("Months Inactive", 0, 12, 3),
        "contacts_count": st.number_input("Service Contacts", 0, 20, 2),
        "late_payments": st.number_input("Late Payments", 0, 10, 1),
        "digital_logins": st.number_input("Digital Logins", 0, 100, 10),
    }
    prediction = {
        "churn_probability": st.slider("Churn Probability", 0.0, 1.0, 0.65),
        "risk_level": st.selectbox("Risk Level", ["Low", "Medium", "High", "Critical"]),
    }
    if st.button("Generate Strategy"):
        strategy = generate_retention_strategy(customer, prediction)
        st.json(strategy)
