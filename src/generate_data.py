"""
Generate realistic fraud detection dataset mirroring Kaggle structure.

This script creates a dataset with:
- Time (seconds from first transaction)
- Amount (transaction value)
- V1-V10 (Simulated PCA-transformed features)
- Class (0 for Legitimate, 1 for Fraud)
- Extremely high imbalance (99.8% legitimate)
"""
import pandas as pd
import numpy as np
import os

def generate_kagglestyle_data(n_samples=5000000, fraud_ratio=0.002, seed=42):
    np.random.seed(seed)
    
    # 1. Generate Time and Amount
    time = np.sort(np.random.uniform(0, 172800, n_samples)) # 2 days in seconds
    amount = np.random.exponential(scale=100, size=n_samples)
    
    # 2. Generate Class (target)
    is_fraud = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])
    
    # 3. Generate V1-V10 (Simulated PCA features)
    # Fraudulent transactions will have different mean/std for these features
    v_features = {}
    for i in range(1, 11):
        # Legitimate: centered around 0
        legit_v = np.random.normal(loc=0, scale=1.0, size=n_samples)
        # Fraud: shifted and higher variance
        fraud_shift = np.random.uniform(-3, 3)
        fraud_v = np.random.normal(loc=fraud_shift, scale=2.0, size=n_samples)
        
        v_features[f'V{i}'] = np.where(is_fraud == 1, fraud_v, legit_v)
        
    # Combine into DataFrame
    df = pd.DataFrame(v_features)
    df['Time'] = time
    df['Amount'] = amount
    df['Class'] = is_fraud
    
    # Add some noise to Amount for fraud
    df.loc[df['Class'] == 1, 'Amount'] = df.loc[df['Class'] == 1, 'Amount'] * np.random.uniform(2, 10, size=sum(is_fraud))
    
    # Reorder columns
    cols = ['Time'] + [f'V{i}' for i in range(1, 11)] + ['Amount', 'Class']
    df = df[cols]
    
    # Split into train/test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print(f"Realistic dataset generated successfully!")
    print(f"Total samples: {n_samples:,}")
    print(f"Fraud count: {sum(is_fraud)} ({fraud_ratio:.2%})")
    print(f"Feature set: Time, Amount, V1-V10")

if __name__ == "__main__":
    generate_kagglestyle_data()
