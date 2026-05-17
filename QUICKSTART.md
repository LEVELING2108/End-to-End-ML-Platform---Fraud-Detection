# Quick Start Guide

This guide helps you get the ML Platform up and running quickly!

## 1️⃣ Prerequisites
- Python 3.9+ installed
- Terminal/Command Prompt access
- ~2GB disk space

## 2️⃣ Initial Setup (One-time)

```bash
# Clone or navigate to project directory
cd ml-platform-tutorial

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3️⃣ Train the Model (One-time)

```bash
# Generate synthetic data
python src/generate_data.py

# Train the model
python src/train_naive.py
```

**Expected output**: Model trained and saved to `models/model.pkl`

## 4️⃣ Run the API

```bash
# Start the API server
uvicorn src.serve_validated:app --reload --port 8000
```

**Expected output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Visit: http://localhost:8000/docs for interactive API docs

## 5️⃣ Test the API (in another terminal)

```bash
# Make sure virtual environment is activated in new terminal
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Test valid transaction (should get is_fraud: false)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 50.0, "hour": 14, "day_of_week": 3, "merchant_category": "grocery"}'

# Test high-risk transaction (should get higher fraud_probability)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": 500.0, "hour": 3, "day_of_week": 1, "merchant_category": "online"}'

# Test invalid transaction (should get HTTP 400)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"amount": -100.0, "hour": 25, "day_of_week": 10, "merchant_category": "invalid"}'
```

## 6️⃣ Run Tests

```bash
# Run data quality and model performance tests
pytest tests/test_data_and_model.py -v

# View test coverage
pytest tests/test_data_and_model.py --cov=src
```

## 7️⃣ Try the Interactive API

Open http://localhost:8000/docs in your browser to:
- See all available endpoints
- Try making predictions directly from the browser
- View request/response schemas
- Test with different inputs

## 📊 Monitor for Drift

```bash
# Test drift detection
python src/monitoring.py
```

## ✅ Verify Everything Works

Run this checklist:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list` shows packages)
- [ ] Data generated (`ls data/` shows train.csv and test.csv)
- [ ] Model trained (`ls models/` shows model.pkl)
- [ ] API running (terminal shows "Uvicorn running...")
- [ ] API responds (`curl http://localhost:8000/health` returns 200)
- [ ] Tests pass (`pytest tests/test_data_and_model.py` shows passed)

## 🐛 Common Issues & Fixes

### "Command not found: python"
- Make sure Python 3.9+ is installed and in PATH
- Try `python3` instead of `python`

### "ModuleNotFoundError: No module named 'src'"
- Make sure you're in the `ml-platform-tutorial` directory
- Make sure virtual environment is activated

### "Port 8000 is already in use"
- Use a different port: `uvicorn src.serve_validated:app --port 8001`
- Or kill the process using port 8000

### "Model is not found"
- Run `python src/train_naive.py` to train the model

## 📚 Next: Explore the Code

1. **Start here**: Read `README.md` for overview
2. **Data generation**: Look at `src/generate_data.py`
3. **Model training**: Look at `src/train_naive.py`
4. **API server**: Look at `src/serve_validated.py`
5. **Validation**: Look at `src/data_validation.py`
6. **Monitoring**: Look at `src/monitoring.py`
7. **Tests**: Look at `tests/test_data_and_model.py`

## 🎓 Learning Path

1. **Basic**: Run the API and make predictions
2. **Intermediate**: Understand the code and modify hyperparameters
3. **Advanced**: Add new features, integrate with real data, deploy to cloud

## 📞 Need Help?

- **API questions**: Check the interactive docs at http://localhost:8000/docs
- **Code questions**: Read the docstrings in each Python file
- **Concepts**: Read the main `README.md` file
- **Full article**: https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/

---

**You're all set! Start with step 4 to run the API.** 🚀
