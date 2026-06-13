"""
Advanced training script with Imbalanced Learning (SMOTE).

This version:
- Handles 99.8% class imbalance using SMOTE
- Logs AUPRC (Area Under Precision-Recall Curve) - crucial for fraud
- Logs full model package to MLflow Model Registry
"""
import pandas as pd
import numpy as np
import pickle
import shap
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import os
import gc

# Set MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("FraudShield_Imbalanced_Learning")

def main():
    print("Loading realistic data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    # Free up memory
    del train_df
    gc.collect()
    
    print(f"Original imbalance: {y_train.value_counts(normalize=True)[1]:.2%} fraud")
    
    # 1. Apply SMOTE to handle imbalance
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Free up memory
    del X_train, y_train
    gc.collect()
    
    print(f"Resampled size: {len(X_train_res):,} samples")
    
    # Memory-optimized parameters
    params = {
        "n_estimators": 50, # Reduced from 100
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": 2 # Limited from -1 to reduce process overhead
    }
    
    with mlflow.start_run(run_name="RandomForest_5M_Optimized"):
        print("Training Random Forest with Imbalanced Learning (Optimized)...")
        model = RandomForestClassifier(**params)
        model.fit(X_train_res, y_train_res)
        
        # 2. Evaluate (using original test data, NOT resampled)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auprc": average_precision_score(y_test, y_prob)
        }
        
        print(f"Metrics: {metrics}")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # 3. Explainability (Using a subset for SHAP to avoid memory crash)
        print("Initializing SHAP (on sample)...")
        explainer = shap.TreeExplainer(model)
        
        # 4. Save and Register
        signature = infer_signature(X_test, y_pred)
        model_info = {"model": model, "encoder": None, "explainer": explainer}
        
        if not os.path.exists("models"): os.makedirs("models")
        with open("models/model_package.pkl", "wb") as f:
            pickle.dump(model_info, f)
            
        mlflow.log_artifact("models/model_package.pkl", artifact_path="model_package")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fraud-model",
            signature=signature,
            registered_model_name="FraudDetectionModel"
        )
        
        print("\nSUCCESS: Advanced Imbalanced Model trained and registered.")

if __name__ == "__main__":
    main()
