# 💳 Customer Churn Prediction & Retention Intelligence System

> **Enterprise-grade ML + GenAI analytics platform** — built for fintech companies like American Express, JPMorgan Chase, Visa, and Mastercard.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
Raw Customer Data (CSV / Hive)
         │
         ▼
  PySpark ETL Pipeline  ──── feature engineering, window functions, joins
         │
         ▼
  SQLite Analytics DB   ──── Hive-style SQL: churn %, revenue trends, at-risk
         │
         ├──────────────────────────────────────────────────┐
         ▼                                                  ▼
  XGBoost Churn Model                              KMeans Segmentation
  (Accuracy 89.8%, AUC 95.2%)                     (4 segments)
         │                                                  │
         └──────────────────────────────────────────────────┘
                             │
                             ▼
                  Claude AI / Rule-Based Engine
                  (Personalized Retention Strategies)
                             │
                             ▼
                   Streamlit Dashboard (6 pages)
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/yourusername/customer-churn-system
cd customer-churn-system
pip install -r requirements.txt

# 2. Run full pipeline (data → spark → SQL → train)
python main.py

# 3. Launch dashboard
streamlit run app/main_dashboard.py

# 4. (Optional) Docker
docker build -t churn-system .
docker run -p 8501:8501 churn-system
```

**GenAI setup (optional):**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```
Without an API key, the system uses a sophisticated rule-based retention engine.

---

## 📁 Project Structure

```
customer-churn-system/
├── data/
│   ├── generate_data.py        # Synthetic fintech dataset generator
│   ├── customers.csv           # 5,000 customer records (auto-generated)
│   └── churn_analytics.db      # SQLite analytics database
├── models/
│   ├── xgb_churn_model.pkl     # Trained XGBoost model
│   ├── kmeans_segments.pkl     # Customer segmentation model
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoders.pkl      # Categorical encoders
│   └── metrics.json            # Model performance metrics
├── app/
│   └── main_dashboard.py       # 6-page Streamlit dashboard
├── sql/
│   └── analytics.py            # SQL analytics module
├── spark_pipeline.py           # PySpark / Pandas ETL pipeline
├── churn_prediction.py         # Prediction module (single + batch)
├── retention_ai.py             # GenAI retention strategy generator
├── train_model.py              # Model training script
├── main.py                     # Full pipeline runner
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Features

| Module | Technology | Description |
|--------|-----------|-------------|
| Data Ingestion | Pandas + NumPy | 5,000 synthetic credit card customers, 25 features |
| Big Data Processing | PySpark (+ Pandas fallback) | ETL, window functions, feature engineering |
| SQL Analytics | SQLite (Hive-style) | 6 production analytics queries |
| Churn Prediction | XGBoost | 89.8% accuracy, 95.2% ROC-AUC |
| Segmentation | KMeans (k=4) | High Value / Low Engagement / Loyal / At-Risk |
| GenAI Advisor | Claude API / Rule-based | Personalized retention strategies |
| Dashboard | Streamlit | 6-page interactive dashboard |

---

## 🎯 Dashboard Pages

1. **Overview** — KPIs, churn distribution, spend analysis, model metrics
2. **Dataset Explorer** — Filterable data table, statistical summary
3. **Churn Predictor** — Real-time single-customer prediction form
4. **Segmentation** — KMeans clusters, PCA visualization, segment profiles
5. **Business Analytics** — SQL-driven insights, revenue at risk, correlation heatmap
6. **AI Retention Advisor** — GenAI-powered personalized retention strategies

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | **89.8%** |
| ROC-AUC | **95.2%** |
| Precision (Churn) | 80% |
| Recall (Churn) | 65% |
| F1-Score | 72% |
| Algorithm | XGBoost (300 estimators) |

**Top churn predictors:** Months Inactive, Total Transaction Count Change, Contacts Count, Late Payments, Digital Logins


