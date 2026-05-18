"""
train_model.py
Trains XGBoost churn prediction model + KMeans segmentation.
Saves models to /models directory.
"""
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, "data", "customers.csv")
MODEL_DIR  = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature list ───────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "months_on_book", "credit_limit", "revolving_balance",
    "avg_open_to_buy", "utilization_ratio", "total_trans_amount",
    "total_trans_count", "avg_trans_amount", "total_amt_change_q4_q1",
    "total_ct_change_q4_q1", "months_inactive_12m", "contacts_count_12m",
    "late_payments", "rewards_redemptions", "digital_logins_30d",
    "dependent_count"
]
CAT_FEATURES = ["gender", "income_bracket", "education_level",
                "marital_status", "card_type"]
TARGET = "churn_label"


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    le_dict = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    feat_cols = NUMERIC_FEATURES + [c + "_enc" for c in CAT_FEATURES]
    X = df[feat_cols]
    y = df[TARGET]
    return df, X, y, feat_cols, le_dict


def train_xgboost(X_train, y_train, X_test, y_test, feat_cols):
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_prob),
        "report":   classification_report(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "feature_importance": dict(zip(feat_cols, model.feature_importances_))
    }
    return model, metrics


def train_segmentation(X_scaled):
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def save_artifacts(model, km, scaler, le_dict, feat_cols):
    joblib.dump(model,    os.path.join(MODEL_DIR, "xgb_churn_model.pkl"))
    joblib.dump(km,       os.path.join(MODEL_DIR, "kmeans_segments.pkl"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le_dict,  os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(feat_cols,os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print("✅ All models saved to /models/")


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Existing","Churned"],
                yticklabels=["Existing","Churned"], ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "screenshots", "confusion_matrix.png"), dpi=100)
    plt.close()


def plot_feature_importance(fi_dict):
    fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    names, vals = zip(*fi)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names[::-1], vals[::-1], color="#0066CC")
    ax.set_title("Top 15 Feature Importances", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "screenshots", "feature_importance.png"), dpi=100)
    plt.close()


if __name__ == "__main__":
    print("🔄 Loading data...")
    df, X, y, feat_cols, le_dict = load_and_preprocess()

    print("🔄 Splitting & scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("🔄 Training XGBoost...")
    model, metrics = train_xgboost(X_train, y_train, X_test, y_test, feat_cols)
    print(f"   Accuracy : {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(metrics["report"])

    print("🔄 Training KMeans segmentation...")
    km, seg_labels = train_segmentation(X_train_sc)

    print("🔄 Saving artifacts...")
    save_artifacts(model, km, scaler, le_dict, feat_cols)

    print("🔄 Saving plots...")
    os.makedirs(os.path.join(BASE, "screenshots"), exist_ok=True)
    plot_confusion_matrix(metrics["conf_matrix"])
    plot_feature_importance(metrics["feature_importance"])

    # Save metrics for dashboard
    import json
    m = {
        "accuracy": round(metrics["accuracy"], 4),
        "roc_auc":  round(metrics["roc_auc"], 4),
        "feature_importance": {k: round(float(v), 6) for k,v in metrics["feature_importance"].items()}
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print("✅ Training complete!")
