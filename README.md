from __future__ import annotations

import json
from pathlib import Path

from src.config import BASE_DIR


def write_readme() -> None:
    readme = """# Customer Churn Prediction & Retention Intelligence System

This repository was rebuilt into a clean, modular customer churn prediction and retention analytics platform.

## Overview

The application generates a realistic synthetic fintech customer dataset, trains an XGBoost model, stores analytics in SQLite, and presents a Streamlit dashboard with real-time prediction and retention guidance.

## Quick Start

1. Create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Initialize the system: `python main.py`
4. Launch the dashboard: `streamlit run app/main.py`

## Project Structure

- app/ - Streamlit dashboard and pages
- src/ - reusable project modules
- data/ - raw and processed customer data
- database/ - SQLite database files
- models/ - trained model and pipeline artifacts
- tests/ - automated tests

## Features

- synthetic data generation with reproducible seed
- one preprocessing pipeline used for training and inference
- XGBoost churn model with metrics output
- KMeans customer segmentation
- SQLite analytics database
- AI retention advisor with rule-based fallback
- Docker support for streamlined deployment

## Commands

- `python main.py`
- `pytest`
- `streamlit run app/main.py`
- `docker build -t customer-churn-system .`

## Notes

The core application does not depend on PySpark and works using Pandas + scikit-learn + XGBoost.
"""
    (BASE_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    write_readme()
