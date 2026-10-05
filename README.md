# Customer Churn Prediction & Retention Intelligence System

A production-style churn analytics platform for fintech customer retention. The system creates a realistic synthetic customer dataset, trains an XGBoost churn model, builds a SQLite analytics database, exposes a Streamlit dashboard, and provides retention guidance with a deterministic fallback engine.

## Overview

This rebuilt application is designed to run reliably on a fresh installation without requiring Spark or a live API. The platform:

- generates a realistic synthetic fintech customer dataset when no dataset exists
- preprocesses features through one canonical sklearn pipeline
- trains a reproducible XGBoost churn model
- segments customers with KMeans
- stores analytics in SQLite
- serves a multi-page Streamlit dashboard
- supports real-time and batch churn prediction
- offers retention recommendations with optional Anthropic Claude support and rule-based fallback

## Architecture

customer-churn-system/
├── app/
│   ├── main.py
│   ├── pages/
│   │   ├── 01_Overview.py
│   │   ├── 02_Data_Explorer.py
│   │   ├── 03_Churn_Predictor.py
│   │   ├── 04_Segmentation.py
│   │   ├── 05_Business_Analytics.py
│   │   └── 06_Retention_Advisor.py
│   └── __init__.py
├── src/
│   ├── analytics/
│   │   └── sql_analytics.py
│   ├── ai/
│   │   └── retention_advisor.py
│   ├── config.py
│   ├── data/
│   │   ├── generator.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── engineering.py
│   └── models/
│       ├── churn_model.py
│       ├── inference.py
│       └── segmentation.py
├── data/
│   ├── raw/
│   └── processed/
├── database/
├── models/
├── scripts/
│   ├── generate_data.py
│   ├── initialize.py
│   └── train.py
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── main.py
├── requirements.txt
├── requirements-dev.txt
└── README.md

## Features

- synthetic fintech customer generation with reproducible seed
- robust preprocessing pipeline shared by training and inference
- XGBoost churn classification with metrics output
- customer segmentation with KMeans and visual analysis
- SQLite analytics with churn KPIs and behavioral trends
- multi-page Streamlit dashboard with safe startup checks
- real-time and batch prediction capabilities
- rule-based AI advisor fallback when no API key is configured

## Tech Stack

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SQLite
- Plotly
- pytest
- Docker

## Installation

### Windows

1. Clone the repo.
2. Create a virtual environment:
   python -m venv .venv
   .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Initialize the project:
   python main.py
5. Launch the app:
   streamlit run app/main.py

### Linux/macOS

1. Clone the repo.
2. Create a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Initialize the project:
   python main.py
5. Launch the app:
   streamlit run app/main.py

## Docker

Build:

docker build -t customer-churn-system .

Run:

docker run -p 8501:8501 customer-churn-system

## Environment Variables

Copy `.env.example` to `.env` and update values as needed.

ANTHROPIC_API_KEY=
DATABASE_PATH=database/churn.db
DATA_PATH=data/processed/customers.csv
MODEL_PATH=models/churn_pipeline.joblib
RANDOM_SEED=42

## Initialization

Run a full setup with:

python main.py

This command will:

- create required directories
- generate the dataset if it does not exist
- preprocess data
- train the XGBoost churn model
- save metrics
- train the segmentation model
- initialize the SQLite database
- validate analytics and save artifacts

## Dashboard Usage

After initialization:

streamlit run app/main.py

The dashboard includes pages for:

- Overview
- Data Explorer
- Churn Predictor
- Customer Segmentation
- Business Analytics
- AI Retention Advisor

## Real-Time Prediction

The churn predictor loads the persisted model and preprocessing pipeline from disk and applies the same transformation logic during inference. No retraining happens at prediction time.

## Batch Prediction

CSV upload is supported in the dashboard workflow for batch scoring and download.

## Model Metrics

Metrics are computed during training and saved to `models/metrics.json`. They are generated from the active training run, not hardcoded values.

## Testing

Run:

pytest

## Troubleshooting

- If the model is missing, run `python main.py`.
- If the database is missing, it will be created automatically during initialization.
- If Anthropic credentials are absent, the retention advisor automatically falls back to the deterministic rule engine.
- If Spark is not installed, the application still works with Pandas and XGBoost.

## Notes

This project intentionally avoids a hard dependency on PySpark. Pandas-based processing is the primary path.
