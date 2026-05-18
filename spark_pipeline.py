"""
spark_pipeline.py
PySpark big-data processing pipeline for customer churn analytics.
Simulates enterprise-grade ETL / feature engineering.
"""
import os, sys

# ── Try importing PySpark; gracefully degrade if not installed ─────
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import DoubleType, IntegerType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "customers.csv")


def get_spark_session():
    """Create or retrieve SparkSession."""
    spark = (SparkSession.builder
             .appName("CustomerChurnAnalytics")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ── Pandas fallback implementations (mirrors Spark logic) ─────────
def pandas_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering using Pandas (mirrors Spark implementation)."""
    df = df.copy()

    # Risk score composite
    df["risk_score"] = (
        df["months_inactive_12m"] * 0.3 +
        df["contacts_count_12m"] * 0.2 +
        df["late_payments"] * 0.3 +
        (1 - df["utilization_ratio"]) * 0.2
    ).round(4)

    # Engagement score
    df["engagement_score"] = (
        df["digital_logins_30d"] * 0.35 +
        df["rewards_redemptions"] * 0.30 +
        df["total_trans_count"] / df["total_trans_count"].max() * 100 * 0.35
    ).round(4)

    # Spend velocity (amount change weighted by count change)
    df["spend_velocity"] = (
        df["total_amt_change_q4_q1"] * df["total_ct_change_q4_q1"]
    ).round(4)

    # Customer lifetime value proxy
    df["clv_proxy"] = (
        df["total_trans_amount"] * df["months_on_book"] / 12
    ).round(2)

    # Percentile rank within card type
    df["trans_pct_rank"] = df.groupby("card_type")["total_trans_amount"] \
        .rank(pct=True).round(4)

    # Rolling risk category
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[0, 0.3, 0.5, 0.7, float("inf")],
        labels=["Low", "Medium", "High", "Critical"]
    )

    return df


def run_spark_pipeline():
    """Full PySpark pipeline with window functions, aggregations, joins."""
    if not SPARK_AVAILABLE:
        print("⚠️  PySpark not installed – running Pandas fallback pipeline.")
        df = pd.read_csv(DATA_PATH)
        result = pandas_feature_engineering(df)
        out = os.path.join(BASE, "data", "customers_engineered.csv")
        result.to_csv(out, index=False)
        print(f"✅ Pandas pipeline complete. Rows: {len(result)}")
        return result

    spark = get_spark_session()
    print("✅ SparkSession created.")

    # ── 1. Load data ──────────────────────────────────────────────
    sdf = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    print(f"   Loaded {sdf.count()} rows, {len(sdf.columns)} columns")

    # ── 2. Basic aggregations ─────────────────────────────────────
    print("   Running aggregations...")
    card_stats = sdf.groupBy("card_type").agg(
        F.count("customer_id").alias("customer_count"),
        F.avg("total_trans_amount").alias("avg_spend"),
        F.sum("churn_label").alias("churned_count"),
        F.avg("credit_limit").alias("avg_credit_limit")
    )
    card_stats.show()

    # ── 3. Window functions ───────────────────────────────────────
    print("   Applying window functions...")
    w_card = Window.partitionBy("card_type").orderBy(F.desc("total_trans_amount"))
    sdf = sdf.withColumn("spend_rank_in_card", F.rank().over(w_card))
    sdf = sdf.withColumn("spend_pct_rank",
        F.percent_rank().over(
            Window.partitionBy("card_type").orderBy("total_trans_amount")
        )
    )

    # ── 4. Feature engineering ────────────────────────────────────
    print("   Engineering features...")
    sdf = sdf.withColumn(
        "risk_score",
        (F.col("months_inactive_12m") * 0.3 +
         F.col("contacts_count_12m") * 0.2 +
         F.col("late_payments") * 0.3 +
         (F.lit(1) - F.col("utilization_ratio")) * 0.2)
    )
    sdf = sdf.withColumn(
        "engagement_score",
        (F.col("digital_logins_30d") * 0.35 +
         F.col("rewards_redemptions") * 0.30 +
         (F.col("total_trans_count") / F.lit(140)) * 100 * 0.35)
    )
    sdf = sdf.withColumn(
        "spend_velocity",
        F.col("total_amt_change_q4_q1") * F.col("total_ct_change_q4_q1")
    )
    sdf = sdf.withColumn(
        "clv_proxy",
        F.col("total_trans_amount") * F.col("months_on_book") / F.lit(12)
    )

    # ── 5. Filtering ──────────────────────────────────────────────
    high_risk = sdf.filter(
        (F.col("months_inactive_12m") >= 3) &
        (F.col("contacts_count_12m") >= 3) &
        (F.col("churn_label") == 1)
    )
    print(f"   High-risk churners: {high_risk.count()}")

    # ── 6. Save ───────────────────────────────────────────────────
    out = os.path.join(BASE, "data", "customers_engineered.csv")
    result_df = sdf.toPandas()
    result_df.to_csv(out, index=False)
    print(f"✅ Spark pipeline complete. Output: {out}")
    spark.stop()
    return result_df


if __name__ == "__main__":
    run_spark_pipeline()
