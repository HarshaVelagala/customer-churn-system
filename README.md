from __future__ import annotations

import json
from pathlib import Path

from src.config import ROOT


def write_readme() -> None:
    content = """# Customer Churn Prediction & Retention Intelligence System

This repository contains a clean, reproducible churn prediction platform built with Python, Streamlit, XGBoost, SQLite, and Plotly.

## Features

- synthetic fintech customer dataset generation
- canonical preprocessing pipeline shared between training and inference
- XGBoost churn model with metrics output
- customer segmentation with KMeans
- SQLite analytics database for KPI reporting
- Streamlit dashboard with multiple pages
- rule-based retention advisor with optional Anthropic support
- Docker support and test suite

## Quick Start

1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Initialize: `python main.py`
4. Run dashboard: `streamlit run app/main.py`

## Commands

- `python main.py`
- `pytest`
- `streamlit run app/main.py`
- `docker build -t customer-churn-system .`

## Notes

The application does not depend on PySpark. It works using Pandas and scikit-learn/XGBoost on a regular local machine.
"""
    (ROOT / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    write_readme()
