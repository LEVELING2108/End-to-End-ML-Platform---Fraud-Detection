# 🧠 Project Memory: FraudShield Enterprise

This file is PRIVATE and ignored by Git. It is for local context only.

## 📍 Current Status: **Production Ready**
The project has been successfully transformed from a basic tutorial script into a commercial-grade MLOps ecosystem.

## 🏗️ Technical Architecture
- **API**: FastAPI secured with `X-API-KEY`. Running on Gunicorn with 4 workers.
- **Frontend**: Streamlit Management Cockpit with SHAP visualization and PDF export.
- **Storage**: PostgreSQL (SQLAlchemy ORM) for persistent audit logs and MLflow metadata.
- **ML Governance**: MLflow Model Registry using "Champion/Challenger" aliasing.
- **Monitoring**: Evidently AI for real-time Data Drift and Data Quality reporting.
- **Infrastructure**: Fully containerized via `docker-compose.yml`.

## 🚀 Enhancements Completed (Resume Highlights)
1. **Architecture Visualization**: Integrated Mermaid.js diagrams into README.
2. **Real-World Data Migration**: Moved to high-dimensional Kaggle-style data (V1-V10) and implemented SMOTE for imbalanced learning (99.8% class imbalance).
3. **Database Persistence**: Implemented a stateful backend using PostgreSQL for permanent audit trails.
4. **Professional CI/CD**: Established a GitHub Actions pipeline with automated model quality gates and Snyk security scanning.

## 🔑 Critical Configurations
- **Security**: Default API Key is `fraud-detection-secret-key`.
- **Ports**: 
  - API: 8000
  - Dashboard: 8501
  - MLflow: 5000
  - Postgres: 5432
- **Model Name**: `FraudDetectionModel`
