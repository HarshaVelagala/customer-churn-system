"""
churn_prediction.py
Loads trained model and makes churn predictions for single customers or batches.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

def load_artifacts():
    model     = joblib.load(os.path.join(MODEL_DIR, "xgb_churn_model.pkl"))
    km        = joblib.load(os.path.join(MODEL_DIR, "kmeans_segments.pkl"))
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le_dict   = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, km, scaler, le_dict, feat_cols


def encode_customer(customer_dict: dict, le_dict: dict) -> dict:
    """Encode categorical fields for a single customer dict."""
    encoded = dict(customer_dict)
    CAT_FEATURES = ["gender","income_bracket","education_level","marital_status","card_type"]
    for col in CAT_FEATURES:
        le = le_dict[col]
        val = str(customer_dict.get(col, le.classes_[0]))
        if val in le.classes_:
            encoded[col + "_enc"] = int(le.transform([val])[0])
        else:
            encoded[col + "_enc"] = 0
    return encoded


def predict_single(customer_dict: dict):
    """
    Predict churn probability for a single customer.
    Returns dict with churn_probability, churn_label, segment, risk_level.
    """
    model, km, scaler, le_dict, feat_cols = load_artifacts()
    encoded = encode_customer(customer_dict, le_dict)
    row = pd.DataFrame([encoded])[feat_cols]
    row = row.fillna(0)

    prob   = float(model.predict_proba(row)[0][1])
    label  = int(model.predict(row)[0])
    scaled = scaler.transform(row)
    seg    = int(km.predict(scaled)[0])

    SEGMENT_NAMES = {
        0: "High Value",
        1: "Low Engagement",
        2: "Loyal Customer",
        3: "At-Risk"
    }
    risk_level = (
        "Critical" if prob >= 0.75 else
        "High"     if prob >= 0.55 else
        "Medium"   if prob >= 0.35 else
        "Low"
    )
    return {
        "churn_probability": round(prob, 4),
        "churn_label":       label,
        "churn_flag":        "Churned" if label else "Existing",
        "segment":           SEGMENT_NAMES.get(seg, f"Segment {seg}"),
        "risk_level":        risk_level
    }


def predict_batch(df: pd.DataFrame):
    """Predict churn for a DataFrame of customers."""
    model, km, scaler, le_dict, feat_cols = load_artifacts()
    CAT_FEATURES = ["gender","income_bracket","education_level","marital_status","card_type"]
    df2 = df.copy()
    for col in CAT_FEATURES:
        le = le_dict[col]
        df2[col+"_enc"] = df2[col].astype(str).apply(
            lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
        )
    X = df2[feat_cols].fillna(0)
    df2["churn_probability"] = model.predict_proba(X)[:, 1].round(4)
    df2["churn_label_pred"]  = model.predict(X)
    seg = km.predict(scaler.transform(X))
    SEGMENT_NAMES = {0:"High Value",1:"Low Engagement",2:"Loyal Customer",3:"At-Risk"}
    df2["segment"] = [SEGMENT_NAMES.get(s, str(s)) for s in seg]
    return df2


if __name__ == "__main__":
    # Demo prediction
    sample = {
        "age": 45, "months_on_book": 36, "credit_limit": 8000,
        "revolving_balance": 500, "avg_open_to_buy": 7500,
        "utilization_ratio": 0.06, "total_trans_amount": 3000,
        "total_trans_count": 45, "avg_trans_amount": 66.7,
        "total_amt_change_q4_q1": 0.6, "total_ct_change_q4_q1": 0.5,
        "months_inactive_12m": 4, "contacts_count_12m": 4,
        "late_payments": 2, "rewards_redemptions": 1,
        "digital_logins_30d": 3, "dependent_count": 2,
        "gender": "Male", "income_bracket": "$40K-$60K",
        "education_level": "Graduate", "marital_status": "Married",
        "card_type": "Blue"
    }
    result = predict_single(sample)
    print("Prediction:", result)
