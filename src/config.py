from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customers.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "customers.csv"
DATABASE_PATH = ROOT / "database" / "churn.db"
MODEL_DIR = ROOT / "models"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def ensure_directories() -> None:
    directories = [
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        ROOT / "database",
        ROOT / "models",
        ROOT / "app" / "pages",
        ROOT / "src" / "data",
        ROOT / "src" / "features",
        ROOT / "src" / "models",
        ROOT / "src" / "analytics",
        ROOT / "src" / "ai",
        ROOT / "scripts",
        ROOT / "tests",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
