"""
Advanced training script with MLflow tracking and Model Registry.

This version logs:
- Hyperparameters
- Performance metrics
- Model artifacts
- SHAP explainer
- Registers the model for production lifecycle management
"""
import pandas as pd
import numpy as np
import pickle
import shap
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)
import os

# Set MLflow tracking URI (local sqlite database)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Fraud_Detection_System")

def main():
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Encode categorical features
    encoder = LabelEncoder()
    train_df["merchant_encoded"] = encoder.fit_transform(train_df["merchant_category"])
    test_df["merchant_encoded"] = encoder.transform(test_df["merchant_category"])
    
    feature_cols = ["amount", "hour", "day_of_week", "merchant_encoded"]
    X_train = train_df[feature_cols]
    y_train = train_df["is_fraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_fraud"]
    
    # Hyperparameters - UPDATED FOR CHALLENGER MODEL
    params = {
        "n_estimators": 200,      # Increased from 100
        "max_depth": 15,          # Increased from 10
        "min_samples_split": 5,   # New constraint
        "random_state": 42
    }
    
    with mlflow.start_run(run_name="RandomForest_Challenger_V2"):
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Predictions & Metrics
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        # Log to MLflow
        print("Logging to MLflow...")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Infer model signature
        signature = infer_signature(X_test, y_pred)
        
        # Log model and encoder
        # We save them as a dictionary/package for easy loading
        model_info = {
            "model": model,
            "encoder": encoder,
            "explainer": explainer
        }
        
        # Save locally first as a temporary artifact
        if not os.path.exists("models"):
            os.makedirs("models")
        with open("models/model_package.pkl", "wb") as f:
            pickle.dump(model_info, f)
            
        mlflow.log_artifact("models/model_package.pkl", artifact_path="model_package")
        
        # Log the sklearn model directly for registry use
        # Note: This version doesn't include the encoder/explainer in the standard MLflow load
        # so we'll prefer loading the full package artifact in the API
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fraud-model",
            signature=signature,
            registered_model_name="FraudDetectionModel"
        )
        
        print("\n" + "="*50)
        print("ADVANCED TRAINING COMPLETE")
        print("="*50)
        print(f"Metrics: {metrics}")
        print("Model registered in MLflow as 'FraudDetectionModel'")
        print("Check MLflow UI (port 5000) for details.")

if __name__ == "__main__":
    main()
