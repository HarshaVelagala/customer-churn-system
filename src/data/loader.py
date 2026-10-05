from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_PATH


def load_customer_data(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path is not None else PROCESSED_DATA_PATH
    if not data_path.exists():
        from src.data.generator import generate_customer_data

        generate_customer_data(output_path=data_path)
    return pd.read_csv(data_path)
