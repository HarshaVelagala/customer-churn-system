"""
sql/analytics.py  –  SQL analytics using SQLite (Hive-style simulation).
Run: python sql/analytics.py
"""
import sqlite3, os, pandas as pd

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "customers.csv")
DB_PATH   = os.path.join(BASE, "data", "churn_analytics.db")

QUERIES = {
    "churn_rate_by_card": """
        SELECT card_type,
               COUNT(*) AS total_customers,
               SUM(churn_label) AS churned,
               ROUND(100.0*SUM(churn_label)/COUNT(*),2) AS churn_pct
        FROM customers
        GROUP BY card_type
        ORDER BY churn_pct DESC
    """,
    "top_spenders": """
        SELECT customer_id, card_type,
               ROUND(total_trans_amount,2) AS spend,
               churn_flag
        FROM customers
        ORDER BY total_trans_amount DESC
        LIMIT 20
    """,
    "monthly_revenue_proxy": """
        SELECT income_bracket,
               ROUND(AVG(total_trans_amount),2) AS avg_monthly_spend,
               COUNT(*) AS customers,
               SUM(churn_label) AS churned
        FROM customers
        GROUP BY income_bracket
        ORDER BY avg_monthly_spend DESC
    """,
    "inactive_users": """
        SELECT customer_id, months_inactive_12m,
               contacts_count_12m, churn_flag
        FROM customers
        WHERE months_inactive_12m >= 3
          AND total_trans_count < 30
        ORDER BY months_inactive_12m DESC
        LIMIT 50
    """,
    "retention_metrics": """
        SELECT
            ROUND(100.0*(SELECT COUNT(*) FROM customers WHERE churn_label=0)
                  / COUNT(*), 2)                          AS retention_rate,
            ROUND(100.0*(SELECT COUNT(*) FROM customers WHERE churn_label=1)
                  / COUNT(*), 2)                          AS churn_rate,
            ROUND(AVG(CASE WHEN churn_label=0 THEN total_trans_amount END),2)
                                                          AS avg_spend_retained,
            ROUND(AVG(CASE WHEN churn_label=1 THEN total_trans_amount END),2)
                                                          AS avg_spend_churned
        FROM customers
    """,
    "high_value_at_risk": """
        SELECT customer_id, card_type,
               ROUND(total_trans_amount,2) AS spend,
               months_inactive_12m, contacts_count_12m, late_payments
        FROM customers
        WHERE churn_label = 1
          AND total_trans_amount > (
              SELECT AVG(total_trans_amount) FROM customers
          )
        ORDER BY total_trans_amount DESC
        LIMIT 25
    """
}


def build_db():
    df = pd.read_csv(DATA_PATH)
    con = sqlite3.connect(DB_PATH)
    df.to_sql("customers", con, if_exists="replace", index=False)
    con.close()
    print(f"✅ SQLite DB built: {DB_PATH}")


def run_queries():
    con = sqlite3.connect(DB_PATH)
    results = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql_query(sql, con)
        results[name] = df
        print(f"\n── {name} ──")
        print(df.to_string(index=False))
    con.close()
    return results


def get_query_result(query_name: str) -> pd.DataFrame:
    """Helper for dashboard: run a named query and return DataFrame."""
    if not os.path.exists(DB_PATH):
        build_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(QUERIES[query_name], con)
    con.close()
    return df


def run_custom_query(sql: str) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        build_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df


if __name__ == "__main__":
    build_db()
    run_queries()
