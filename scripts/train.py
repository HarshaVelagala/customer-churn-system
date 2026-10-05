from __future__ import annotations

from src.data.generator import generate_customer_data


if __name__ == "__main__":
    generate_customer_data(5000, seed=42)
    print("Synthetic customer dataset generated.")
