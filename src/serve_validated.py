"""
Serve fraud detection model with input validation.

This version adds data validation BEFORE making predictions:
- Invalid inputs are rejected with HTTP 400 and clear error messages
- Valid inputs are processed and predictions returned
"""
import pickle
import numpy as np
import mlflow
import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Dict, List
from src.data_validation import validate_transaction

# API Security Configuration
API_KEY_NAME = "X-API-KEY"
API_KEY = os.getenv("FRAUD_API_KEY", "fraud-detection-secret-key") # Default for demo
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate API Key"
        )

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "FraudDetectionModel"
MODEL_ALIAS = "champion"

def load_model_package():
    """
    Load the model package from MLflow Registry or local fallback.
    """
    # 1. Try loading from MLflow
    try:
        print(f"Attempting to load '{MODEL_ALIAS}' model from MLflow Registry...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load the specific package artifact from the champion version
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run_id = model_version.run_id
        
        # Download the custom package artifact
        artifact_path = "model_package/model_package.pkl"
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        
        with open(local_path, "rb") as f:
            package = pickle.load(f)
        print(f"Successfully loaded '{MODEL_NAME}' (v{model_version.version}) from MLflow!")
        return package["model"], package["encoder"], package["explainer"]
    
    except Exception as e:
        print(f"MLflow load failed: {e}")
        print("Falling back to local 'models/model_package.pkl'...")
        
        # 2. Local Fallback
        if os.path.exists("models/model_package.pkl"):
            with open("models/model_package.pkl", "rb") as f:
                package = pickle.load(f)
                return package["model"], package["encoder"], package["explainer"]
        else:
            raise RuntimeError("No model found in MLflow or local storage!")

# Load model package at startup
print("Initializing model...")
model, encoder, explainer = load_model_package()

app = FastAPI(
    title="Fraud Detection API (Commercial Grade)",
    description="""
    Commercial-grade Fraud detection API.
    
    Features:
    - **MLflow Model Management**: Loads the 'Champion' model from the registry.
    - **SHAP Explainability**: Provides human-readable reasons for every prediction.
    - **Data Validation**: Rejects invalid inputs before they reach the model.
    """,
    version="4.0.0"
)


class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount (must be positive)", example=150.00)
    hour: int = Field(..., description="Hour of day (0-23)", example=14)
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", example=3)
    merchant_category: str = Field(..., description="Merchant type", example="online")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    validation_passed: bool = True


class ExplanationResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    base_value: float
    shap_values: Dict[str, float]
    top_contributors: List[str]
    summary: str


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction, api_key: str = Depends(get_api_key)):
    """
    Predict whether a transaction is fraudulent.
    """
    data = tx.dict()
    
    # VALIDATE INPUT
    validation = validate_transaction(data)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"message": "Validation failed", "errors": validation["errors"]})
    
    # Encode and predict
    data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    X = [[data["amount"], data["hour"], data["day_of_week"], data["merchant_encoded"]]]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4)
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(tx: Transaction, api_key: str = Depends(get_api_key)):
    """
    Get SHAP explanation for a prediction.
    """
    data = tx.dict()
    
    # VALIDATE INPUT
    validation = validate_transaction(data)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={"message": "Validation failed", "errors": validation["errors"]})
    
    # Prepare features
    merchant_encoded = encoder.transform([data["merchant_category"]])[0]
    feature_names = ["amount", "hour", "day_of_week", "merchant_category"]
    X_explain = np.array([[data["amount"], data["hour"], data["day_of_week"], merchant_encoded]])
    
    # Get SHAP values
    # For TreeExplainer, it returns [samples, features, classes] or [classes][samples, features]
    shap_values_raw = explainer.shap_values(X_explain)
    
    # Random Forest in SHAP usually returns a list of two arrays (one for each class)
    # We want class 1 (Fraud)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1][0]
        base_value = explainer.expected_value[1]
    else:
        # Some versions return a single array where the last dim is classes
        shap_values = shap_values_raw[0, :, 1]
        base_value = explainer.expected_value[1]

    # Map SHAP values to feature names
    explanation_map = {
        "amount": float(shap_values[0]),
        "hour": float(shap_values[1]),
        "day_of_week": float(shap_values[2]),
        "merchant_category": float(shap_values[3])
    }
    
    # Sort contributors
    sorted_contributors = sorted(explanation_map.items(), key=lambda x: abs(x[1]), reverse=True)
    top_contributors = [f"{k} ({'+' if v > 0 else ''}{v:.4f})" for k, v in sorted_contributors]
    
    # Prediction
    prob = model.predict_proba(X_explain)[0][1]
    is_fraud = prob > 0.5
    
    # Generate summary
    top_feat, top_val = sorted_contributors[0]
    impact = "increasing" if top_val > 0 else "decreasing"
    summary = f"The {top_feat} was the primary factor {impact} the fraud risk for this transaction."

    return ExplanationResponse(
        is_fraud=bool(is_fraud),
        fraud_probability=round(float(prob), 4),
        base_value=float(base_value),
        shap_values=explanation_map,
        top_contributors=top_contributors,
        summary=summary
    )


@app.get("/health")
def health():
    return {"status": "healthy", "explainability": "enabled"}


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API (Validated + Explainable)",
        "version": "3.0.0",
        "endpoints": ["/predict", "/explain", "/docs"]
    }
