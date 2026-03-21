# End-to-End ML Platform - Fraud Detection

A complete, production-ready machine learning platform built locally with experiment tracking, model versioning, feature management, data validation, monitoring, and CI/CD automation. Based on the [FreeCodeCamp article](https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/).

## 📋 Project Overview

This project demonstrates best practices for building ML systems that go beyond training a model in a notebook. It covers:

- **Experiment Tracking** (MLflow): Log parameters, metrics, and artifacts for reproducible results
- **Model Registry** (MLflow): Version models with aliases and manage deployment lifecycle  
- **Feature Store** (Feast): Ensure consistent features between training and serving
- **Data Validation** (Great Expectations): Validate input data before predictions
- **Monitoring** (Evidently): Detect data drift and model performance decay
- **CI/CD** (GitHub Actions): Automated testing and safe deployments
- **Containerization** (Docker): Environment consistency everywhere

## 🎯 Use Case: Credit Card Fraud Detection

We predict whether a credit card transaction is fraudulent based on:
- **amount**: Transaction amount in dollars
- **hour**: Hour of day (0-23)
- **day_of_week**: Day of week (0=Monday, 6=Sunday)
- **merchant_category**: Type of merchant (grocery, restaurant, retail, online, travel)

## 📁 Project Structure

```
ml-platform-tutorial/
├── data/                          # Training and test datasets
│   ├── train.csv                 # Training data (~8,000 transactions)
│   └── test.csv                  # Test data (~2,000 transactions)
├── models/                        # Saved model files
│   └── model.pkl                 # Trained Random Forest + encoder
├── src/                           # Source code
│   ├── generate_data.py          # Generate synthetic data
│   ├── train_naive.py            # Train the model
│   ├── serve_naive.py            # Basic API server
│   ├── serve_validated.py        # API with validation
│   ├── data_validation.py        # Input validation rules
│   └── monitoring.py             # Drift detection & monitoring
├── feature_repo/                  # Feast feature store
│   ├── feature_store.yaml        # Feast configuration
│   └── features.py               # Feature definitions
├── tests/                         # Test files
│   ├── test_data_and_model.py   # Data quality & model perf tests
│   └── test_api.py               # API endpoint tests
├── .github/workflows/             # CI/CD pipelines
│   └── ci.yml                    # GitHub Actions workflow
├── Dockerfile                     # Container configuration
├── .dockerignore                  # Files to exclude from Docker
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Docker Desktop (optional)
- Git (for version control)

### Step 1: Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Generate Data & Train Model

```bash
# Generate synthetic fraud detection dataset
python src/generate_data.py

# Train the initial model
python src/train_naive.py
```

You should see output showing model performance:
```
==================================================
MODEL EVALUATION
==================================================

Accuracy:  0.9820
Precision: 0.7273
Recall:    0.6154
F1-score:  0.6667
```

### Step 4: Run the API

```bash
uvicorn src.serve_validated:app --reload --host 0.0.0.0 --port 8000
```

Visit the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 5: Test the API

```bash
# Valid transaction (low risk)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}'

# High-risk transaction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 500.0, "hour": 3, "day_of_week": 1, "merchant_category": "online"}'

# Invalid transaction (will be rejected)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": -500.0, "hour": 25, "day_of_week": 10, "merchant_category": "unknown"}'
```

## 🔧 Advanced Features

### 1. Data Validation

Test the data validation module:
```bash
python src/data_validation.py
```

The API automatically rejects invalid inputs with clear error messages:
- Negative amounts
- Invalid hours (outside 0-23)
- Invalid days (outside 0-6)
- Unknown merchant categories

### 2. Drift Detection

Monitor for data drift:
```bash
python src/monitoring.py
```

Tests several scenarios:
- Similar data (minimal drift)
- Amount inflation (2x multiplier)
- Time shift (late-night transactions)

### 3. Run Tests

```bash
# Data quality and model performance tests
pytest tests/test_data_and_model.py -v

# With coverage report
pytest tests/test_data_and_model.py -v --cov=src
```

### 4. Docker Containerization

```bash
# Build the Docker image
docker build -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api

# Test the container
curl http://localhost:8000/health
```

## 📊 Architecture Components

### MLflow (Experiment Tracking & Model Registry)
Tracks every training run with:
- **Parameters**: Hyperparameters (n_estimators, max_depth, etc.)
- **Metrics**: Performance metrics (accuracy, precision, recall, F1)
- **Artifacts**: Trained models and encoders
- **Model Registry**: Version management and aliasing

Start MLflow server:
```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Access MLflow UI: http://localhost:5000

### Feast (Feature Store)
Ensures feature consistency between training and serving:
- Single source of truth for feature computation
- Point-in-time correct joins for training data
- Low-latency online serving for predictions

Initialize and materialize features:
```bash
cd feature_repo
feast apply
feast materialize-incremental <timestamp>
cd ..
```

### Great Expectations (Data Validation)
Defines and validates data quality rules:
- Amount must be positive and < $50,000
- Hour must be 0-23
- Day must be 0-6
- Merchant category must be from known set
- No null values in critical columns

### Evidently (Monitoring)
Detects data drift over time:
- Compares production data against training data
- Uses Kolmogorov-Smirnov (KS) test for drift detection
- Generates alerts when drift exceeds threshold
- Produces HTML reports with visualizations

### GitHub Actions (CI/CD)
Automated pipeline that:
1. Installs dependencies
2. Generates training data
3. Trains the model
4. Runs data quality tests
5. Builds Docker image
6. Tests the containerized API

## 🛡️ Production Readiness Checklist

- [x] **Reproducibility**: Every experiment logged with parameters and metrics
- [x] **Model Versioning**: Models stored in registry with aliases
- [x] **Feature Consistency**: Features defined once, used everywhere
- [x] **Data Validation**: Invalid inputs rejected with clear errors
- [x] **Monitoring**: Drift detection and alerting
- [x] **Containerization**: Docker for environment consistency
- [x] **Testing**: Unit tests for data, model, and API
- [x] **CI/CD**: Automated testing and deployment
- [x] **Documentation**: Clear setup and usage instructions

## 📈 Next Steps for Production

### Scale to Cloud
- Deploy to AWS ECS, Google Cloud Run, or Azure Container Instances
- Use managed services for MLflow and Feast backends
- Integrate with cloud monitoring (CloudWatch, Stackdriver, etc.)

### Advanced Monitoring
- Connect Evidently to Slack or PagerDuty for alerts
- Set up Prometheus + Grafana dashboards
- Implement OpenTelemetry for distributed tracing

### Continuous Training
- Schedule periodic retraining when drift is detected
- Use canary deployments to safely roll out new models
- Implement A/B testing for model comparison

### Model Explainability
- Add SHAP or LIME for prediction explanations
- Track model fairness and bias metrics
- Create audit trails for model decisions

## 🔗 Resources & References

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Feast Documentation](https://docs.feast.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Great Expectations](https://greatexpectations.io/)
- [Evidently AI](https://docs.evidentlyai.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

## 📝 Key Learnings

### Problem 1: No Experiment Tracking
**Issue**: Unable to reproduce or compare experiments
**Solution**: Use MLflow to log all parameters, metrics, and artifacts

### Problem 2: No Model Versioning
**Issue**: Can't roll back to previous models if something breaks
**Solution**: Use MLflow Model Registry with aliases (@champion, @challenger)

### Problem 3: Training-Serving Skew
**Issue**: Features computed differently in training vs production
**Solution**: Use Feast feature store for consistent definitions

### Problem 4: No Data Validation
**Issue**: API accepts garbage input and produces invalid predictions
**Solution**: Use Great Expectations to validate all inputs

### Problem 5: Silent Model Decay
**Issue**: Model performance degrades over time without anyone noticing
**Solution**: Use Evidently to monitor for data and concept drift

### Problem 6: Risky Deployments
**Issue**: Manual deployment process with no automated testing
**Solution**: Use GitHub Actions for automated testing and safe rollouts

## 💡 Tips & Best Practices

1. **Always Track Experiments**: Never train without logging to MLflow
2. **Version Everything**: Models, datasets, and feature definitions
3. **Validate Early**: Reject bad data at the API boundary
4. **Monitor Continuously**: Set up alerts for drift and performance decay
5. **Test Thoroughly**: Automated tests catch regressions before production
6. **Document Well**: Clear runbooks help during incidents
7. **Automate Safely**: Use staging environments and canary deployments

## 🐛 Troubleshooting

### "Model not found" error
```bash
# Make sure you've trained the model first
python src/train_naive.py
```

### "Port 8000 already in use"
```bash
# Use a different port
uvicorn src.serve_validated:app --port 8001
```

### "KeyError: 'merchant_category'"
```bash
# Make sure your input JSON has all required fields
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 50.0,
    "hour": 14,
    "day_of_week": 3,
    "merchant_category": "online"
  }'
```

### Tests fail with "file not found"
```bash
# Make sure data has been generated
python src/generate_data.py

# Make sure model has been trained
python src/train_naive.py
```

## 📄 License

This project is based on the FreeCodeCamp tutorial and is available for educational and learning purposes.

## 🤝 Contributing

Feel free to:
- Report bugs and issues
- Suggest improvements
- Add new features (e.g., more validation rules, additional monitoring)
- Improve documentation

## ❓ FAQ

**Q: Can I use this for production fraud detection?**
A: This is a tutorial project with synthetic data. For production, you'd need real labeled data, more sophisticated models, and proper data governance.

**Q: Do I need Docker?**
A: No, Docker is optional. You can run everything with just Python and virtual environments. Docker is useful for deployment consistency.

**Q: How do I update the model?**
A: Run `python src/train_naive.py` to retrain, then the API will automatically use the updated trained model.

**Q: Can I use different ML models?**
A: Yes! The same framework works with XGBoost, LightGBM, Neural Networks, etc. Just update `train_naive.py`.

**Q: How do I integrate with my own data?**
A: Replace `generate_data.py` with code that reads your data, update validation rules in `data_validation.py`, and retrain the model.

---

**Built with ❤️ for the ML community**

Based on: [How to Build an End-to-End ML Platform Locally: From Experiment Tracking to CI/CD](https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/)
# END_TO_END_ML_PLATFORM
