from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATABASE_PATH, PROCESSED_DATA_PATH


def ensure_database(data_path: str | Path | None = None, db_path: str | Path | None = None) -> Path:
    source = Path(data_path) if data_path is not None else PROCESSED_DATA_PATH
    target = Path(db_path) if db_path is not None else DATABASE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    connection = sqlite3.connect(target)
    df.to_sql("customers", connection, if_exists="replace", index=False)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers (churn_label)")
    connection.commit()
    connection.close()
    return target


def query_database(db_path: str | Path | None = None, query: str = "SELECT * FROM customers") -> pd.DataFrame:
    target = Path(db_path) if db_path is not None else DATABASE_PATH
    connection = sqlite3.connect(target)
    result = pd.read_sql_query(query, connection)
    connection.close()
    return result
