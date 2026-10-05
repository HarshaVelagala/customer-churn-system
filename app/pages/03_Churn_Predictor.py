from __future__ import annotations

import streamlit as st

from src.models.churn_model import predict_single_record


def app() -> None:
    st.title("Churn Predictor")
    record = {
        "age": st.number_input("Age", 21, 90, 40),
        "gender": st.selectbox("Gender", ["Male", "Female"]),
        "income_bracket": st.selectbox("Income Bracket", ["<$40K", "$40K-$60K", "$60K-$80K", "$80K-$120K", "$120K+"]),
        "card_type": st.selectbox("Card Type", ["Blue", "Silver", "Gold", "Platinum"]),
        "education_level": st.selectbox("Education Level", ["High School", "Some College", "Graduate", "Post-Graduate"]),
        "marital_status": st.selectbox("Marital Status", ["Married", "Single", "Divorced"]),
        "credit_limit": st.number_input("Credit Limit", 1000.0, 50000.0, 8000.0),
        "months_on_book": st.number_input("Months on Book", 6, 240, 36),
        "months_inactive": st.number_input("Months Inactive", 0, 12, 2),
        "contacts_count": st.number_input("Contacts Count", 0, 20, 1),
        "total_transaction_count": st.number_input("Transaction Count", 0, 500, 75),
        "total_transaction_amount": st.number_input("Transaction Amount", 0.0, 100000.0, 5000.0),
        "transaction_count_change": st.number_input("Transaction Count Change", -1.0, 3.0, 0.7),
        "transaction_amount_change": st.number_input("Transaction Amount Change", -1.0, 4.0, 0.8),
        "utilization_ratio": st.number_input("Utilization Ratio", 0.0, 1.0, 0.25),
        "revolving_balance": st.number_input("Revolving Balance", 0.0, 25000.0, 1800.0),
        "late_payments": st.number_input("Late Payments", 0, 10, 1),
        "digital_logins": st.number_input("Digital Logins", 0, 100, 12),
        "online_transactions": st.number_input("Online Transactions", 0, 100, 8),
        "customer_service_calls": st.number_input("Customer Service Calls", 0, 20, 2),
        "dependent_count": st.number_input("Dependents", 0, 10, 1),
    }

    if st.button("Predict Churn"):
        try:
            result = predict_single_record(record)
            st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            st.metric("Risk Level", result["risk_level"])
            st.metric("Prediction", "Churned" if result["prediction"] == 1 else "Existing")
            st.json(result)
        except FileNotFoundError as exc:
            st.warning(str(exc))
