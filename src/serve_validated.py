import pickle
import numpy as np
import mlflow
import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Dict, List

# API Security Configuration
API_KEY_NAME = "X-API-KEY"
API_KEY = os.getenv("FRAUD_API_KEY", "fraud-detection-secret-key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate API Key")

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "FraudDetectionModel"
MODEL_ALIAS = "champion"

def load_model_package():
    try:
        print(f"Attempting to load '{MODEL_ALIAS}' model from MLflow Registry...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        run_id = model_version.run_id
        artifact_path = "model_package/model_package.pkl"
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        with open(local_path, "rb") as f:
            package = pickle.load(f)
        print(f"Successfully loaded '{MODEL_NAME}' (v{model_version.version}) from MLflow!")
        return package["model"], package["explainer"]
    except Exception as e:
        print(f"MLflow load failed: {e}. Falling back to local...")
        if os.path.exists("models/model_package.pkl"):
            with open("models/model_package.pkl", "rb") as f:
                package = pickle.load(f)
                return package["model"], package["explainer"]
        else:
            raise RuntimeError("No model found!")

# Load model at startup
model, explainer = load_model_package()

app = FastAPI(
    title="FraudShield Enterprise API (High-Dimensional)",
    description="Secured API for Real-time Fraud Detection on PCA-transformed financial features.",
    version="5.0.0"
)

class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since first transaction", example=450.0)
    Amount: float = Field(..., description="Transaction value", example=120.50)
    V1: float = Field(0.0)
    V2: float = Field(0.0)
    V3: float = Field(0.0)
    V4: float = Field(0.0)
    V5: float = Field(0.0)
    V6: float = Field(0.0)
    V7: float = Field(0.0)
    V8: float = Field(0.0)
    V9: float = Field(0.0)
    V10: float = Field(0.0)

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float

class ExplanationResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    shap_values: Dict[str, float]
    summary: str

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction, api_key: str = Depends(get_api_key)):
    data = tx.dict()
    # No validation for now as features are PCA-transformed
    X = [list(data.values())]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return PredictionResponse(is_fraud=bool(pred), fraud_probability=round(float(prob), 4))

@app.post("/explain", response_model=ExplanationResponse)
def explain(tx: Transaction, api_key: str = Depends(get_api_key)):
    data = tx.dict()
    X = np.array([list(data.values())])
    shap_values_raw = explainer.shap_values(X)
    
    if isinstance(shap_values_raw, list):
        sv = shap_values_raw[1][0]
    else:
        sv = shap_values_raw[0, :, 1] if shap_values_raw.ndim == 3 else shap_values_raw[0]

    explanation_map = {k: float(v) for k, v in zip(data.keys(), sv)}
    prob = model.predict_proba(X)[0][1]
    
    return ExplanationResponse(
        is_fraud=bool(prob > 0.5),
        fraud_probability=round(float(prob), 4),
        shap_values=explanation_map,
        summary="SHAP explanation for high-dimensional feature set."
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model_type": "SMOTE-Optimized Random Forest"}
