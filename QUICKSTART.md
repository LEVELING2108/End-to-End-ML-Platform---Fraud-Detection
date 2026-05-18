# 🚀 FraudShield Quick Start Guide

This guide will get you from a fresh clone to a fully operational, self-healing ML platform in minutes.

## 1️⃣ Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (Recommended for Production Mode)
- **Git**

## 2️⃣ The "One-Command" Launch (Recommended)

The easiest way to run the entire platform (API, Dashboard, and MLflow) is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com/LEVELING2108/End-to-End-ML-Platform---Fraud-Detection.git
cd End-to-End-ML-Platform---Fraud-Detection

# Launch the full enterprise stack
docker-compose up --build
```

**What happens next?**
- The system builds a multi-worker production container.
- It starts the **Fraud API** (Port 8000), **Dashboard** (Port 8501), and **MLflow** (Port 5000).
- You are ready to go!

---

## 3️⃣ Manual Development Setup (Local)

If you prefer to run services individually for development:

### A. Environment Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### B. Initialize Data & Model
```bash
# 1. Generate synthetic fraud data
python src/generate_data.py

# 2. Train advanced model with MLflow tracking
python src/train_advanced.py
```

### C. Launch Services (Individual Terminals)
```bash
# Terminal 1: MLflow Tracking Server
python -m mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///mlflow.db

# Terminal 2: Secured Fraud API
python -m uvicorn src.serve_validated:app --host localhost --port 8000

# Terminal 3: Management Cockpit
python -m streamlit run src/dashboard.py --server.port 8501
```

---

## 4️⃣ Key Operations

### ⚖️ Analyze & Explain
1. Open the **Management Cockpit** (http://localhost:8501).
2. Enter transaction details (e.g., Amount: $5000, Hour: 3 AM).
3. View the **SHAP Force Plot** to see the logic.
4. **Download the PDF Report** for your audit trail.

### 🛡️ Test Security
Every API request requires the `X-API-KEY`.
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-KEY: fraud-detection-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}'
```

### 🔄 Trigger Self-Healing
To see the platform autonomously detect drift and retrain:
```bash
$env:PYTHONPATH="."; python src/orchestrator.py
```

---

## ✅ Final Health Checklist
- [ ] **Dashboard** loads at http://localhost:8501
- [ ] **MLflow** shows "FraudDetectionModel" at http://localhost:5000
- [ ] **API** shows "Uvicorn running" on port 8000
- [ ] **PDF Export** generates a professional report without errors
- [ ] **Audit Log** updates dynamically in the dashboard

---

## 📞 Support & Documentation
- **Detailed Overview**: See [README.md](README.md)
- **API Docs**: Interactive Swagger at http://localhost:8000/docs
- **Original Base**: [FreeCodeCamp MLOps Tutorial](https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/)
