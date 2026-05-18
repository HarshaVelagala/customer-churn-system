"""
main.py – Project entry point
Runs the full pipeline: data generation → spark processing → SQL setup → model training
"""
import os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

def run(script, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script], cwd=BASE)
    if result.returncode != 0:
        print(f"⚠️  {label} completed with warnings.")

if __name__ == "__main__":
    print("\n💳  Customer Churn Intelligence System")
    print("    Enterprise Analytics Pipeline v1.0\n")

    run(os.path.join(BASE, "data", "generate_data.py"),   "1/4 · Generating synthetic dataset")
    run(os.path.join(BASE, "spark_pipeline.py"),           "2/4 · Running Spark / Pandas pipeline")
    run(os.path.join(BASE, "sql", "analytics.py"),         "3/4 · Building SQL analytics database")
    run(os.path.join(BASE, "train_model.py"),              "4/4 · Training ML models")

    print("\n" + "="*55)
    print("  ✅ Pipeline complete!")
    print("  Launch dashboard:")
    print("  streamlit run app/main_dashboard.py")
    print("="*55 + "\n")
