from __future__ import annotations

import json
from pathlib import Path
import os

from src.config import BASE_DIR


def build_project_structure() -> None:
    dirs = [
        BASE_DIR / "app" / "pages",
        BASE_DIR / "src" / "data",
        BASE_DIR / "src" / "features",
        BASE_DIR / "src" / "models",
        BASE_DIR / "src" / "analytics",
        BASE_DIR / "src" / "ai",
        BASE_DIR / "data" / "raw",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "database",
        BASE_DIR / "models",
        BASE_DIR / "tests",
        BASE_DIR / "scripts",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    build_project_structure()
