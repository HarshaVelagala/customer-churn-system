"""
Synthetic Financial Dataset Generator
Generates realistic credit card customer data for churn prediction.
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_customer_data(n=5000):
    """Generate realistic credit card customer dataset."""
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n + 1)]
    ages = np.random.normal(42, 12, n).clip(21, 75).astype(int)
    genders = np.random.choice(["Male", "Female"], n, p=[0.52, 0.48])
    income_brackets = np.random.choice(
        ["<$40K", "$40K-$60K", "$60K-$80K", "$80K-$120K", "$120K+"],
        n, p=[0.15, 0.25, 0.30, 0.20, 0.10]
    )
    education = np.random.choice(
        ["High School", "Some College", "Graduate", "Post-Graduate", "Uneducated"],
        n, p=[0.15, 0.20, 0.35, 0.25, 0.05]
    )
    marital_status = np.random.choice(["Married", "Single", "Divorced"], n, p=[0.55, 0.35, 0.10])
    dependents = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.20, 0.15, 0.30, 0.25, 0.07, 0.03])
    card_types = np.random.choice(["Blue", "Silver", "Gold", "Platinum"], n, p=[0.53, 0.25, 0.15, 0.07])
    months_on_book = np.random.normal(36, 18, n).clip(6, 120).astype(int)
    credit_limit = np.random.lognormal(9.5, 0.8, n).clip(1000, 35000).round(2)
    revolving_bal = (credit_limit * np.random.beta(1.5, 4, n)).round(2)
    avg_open_to_buy = (credit_limit - revolving_bal).round(2)
    utilization = (revolving_bal / credit_limit).round(4)
    total_trans_amt = np.random.lognormal(8.0, 0.9, n).clip(500, 25000).round(2)
    total_trans_ct = np.random.normal(65, 30, n).clip(10, 140).astype(int)
    avg_trans_amt = (total_trans_amt / total_trans_ct).round(2)
    total_amt_chng = np.random.normal(0.75, 0.25, n).clip(0.0, 3.0).round(4)
    total_ct_chng = np.random.normal(0.70, 0.20, n).clip(0.0, 2.5).round(4)
    months_inactive = np.random.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03])
    contacts_count = np.random.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.03, 0.02])
    late_payments = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.55, 0.20, 0.12, 0.07, 0.04, 0.02])
    rewards_redemptions = np.random.poisson(3, n).clip(0, 20)
    digital_logins = np.random.poisson(8, n).clip(0, 40)
    
    # Churn logic: higher probability for inactive, low trans, high contacts, low utilization
    churn_score = (
        0.15 * (months_inactive / 6) +
        0.20 * (1 - total_ct_chng / 2.5) +
        0.15 * (contacts_count / 6) +
        0.10 * (late_payments / 5) +
        0.10 * (1 - rewards_redemptions / 20) +
        0.10 * (1 - digital_logins / 40) +
        0.10 * (1 - total_amt_chng / 3) +
        0.10 * np.random.uniform(0, 1, n)
    )
    churn_label = (churn_score > 0.45).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "income_bracket": income_brackets,
        "education_level": education,
        "marital_status": marital_status,
        "dependent_count": dependents,
        "card_type": card_types,
        "months_on_book": months_on_book,
        "credit_limit": credit_limit,
        "revolving_balance": revolving_bal,
        "avg_open_to_buy": avg_open_to_buy,
        "utilization_ratio": utilization,
        "total_trans_amount": total_trans_amt,
        "total_trans_count": total_trans_ct,
        "avg_trans_amount": avg_trans_amt,
        "total_amt_change_q4_q1": total_amt_chng,
        "total_ct_change_q4_q1": total_ct_chng,
        "months_inactive_12m": months_inactive,
        "contacts_count_12m": contacts_count,
        "late_payments": late_payments,
        "rewards_redemptions": rewards_redemptions,
        "digital_logins_30d": digital_logins,
        "churn_label": churn_label,
        "churn_flag": np.where(churn_label == 1, "Churned", "Existing")
    })
    return df

if __name__ == "__main__":
    df = generate_customer_data(5000)
    out_path = os.path.join(os.path.dirname(__file__), "customers.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset saved: {out_path} | Shape: {df.shape}")
    print(f"Churn Rate: {df['churn_label'].mean():.2%}")
