from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customers.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "customers.csv"
DATABASE_PATH = BASE_DIR / "database" / "churn.db"
MODEL_DIR = BASE_DIR / "models"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def ensure_directories() -> None:
    for path in [DATA_DIR / "raw", DATA_DIR / "processed", BASE_DIR / "database", MODEL_DIR, BASE_DIR / "app" / "pages"]:
        path.mkdir(parents=True, exist_ok=True)
