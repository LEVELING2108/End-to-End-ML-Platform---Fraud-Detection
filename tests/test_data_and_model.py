"""
Tests for data quality and model performance.

These tests run in CI/CD to ensure:
1. Data meets quality requirements
2. Model meets performance thresholds
3. No regressions are introduced

Run with: pytest tests/test_data_and_model.py -v
"""
import pandas as pd
import pickle
import pytest
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TestDataQuality:
    """Tests for training data quality."""
    
    @pytest.fixture
    def train_data(self):
        if not os.path.exists("data/train.csv"):
            pytest.skip("data/train.csv not found")
        return pd.read_csv("data/train.csv")
    
    @pytest.fixture
    def test_data(self):
        if not os.path.exists("data/test.csv"):
            pytest.skip("data/test.csv not found")
        return pd.read_csv("data/test.csv")
    
    def test_train_data_has_expected_columns(self, train_data):
        """Training data must have all required columns."""
        required_columns = {"amount", "hour", "day_of_week", "merchant_category", "is_fraud"}
        actual_columns = set(train_data.columns)
        missing = required_columns - actual_columns
        assert not missing, f"Missing columns: {missing}"
    
    def test_train_data_not_empty(self, train_data):
        """Training data must have rows."""
        assert len(train_data) > 0, "Training data is empty"
        assert len(train_data) >= 1000, f"Training data too small: {len(train_data)} rows"
    
    def test_no_negative_amounts(self, train_data):
        """Transaction amounts must be non-negative."""
        negative_count = (train_data["amount"] < 0).sum()
        assert negative_count == 0, f"Found {negative_count} negative amounts"
    
    def test_amounts_reasonable(self, train_data):
        """Transaction amounts should be within reasonable bounds."""
        max_amount = train_data["amount"].max()
        assert max_amount <= 100000, f"Max amount {max_amount} exceeds reasonable limit"
    
    def test_hours_valid(self, train_data):
        """Hours must be 0-23."""
        invalid = train_data[(train_data["hour"] < 0) | (train_data["hour"] > 23)]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid hours"
    
    def test_days_valid(self, train_data):
        """Days of week must be 0-6."""
        invalid = train_data[(train_data["day_of_week"] < 0) | (train_data["day_of_week"] > 6)]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid days"
    
    def test_merchant_categories_valid(self, train_data):
        """Merchant categories must be from known set."""
        valid_categories = {"grocery", "restaurant", "retail", "online", "travel"}
        actual_categories = set(train_data["merchant_category"].unique())
        invalid = actual_categories - valid_categories
        assert not invalid, f"Invalid merchant categories: {invalid}"
    
    def test_fraud_ratio_reasonable(self, train_data):
        """Fraud ratio should be realistic (between 0.1% and 50%)."""
        fraud_ratio = train_data["is_fraud"].mean()
        assert 0.001 <= fraud_ratio <= 0.5, f"Fraud ratio {fraud_ratio:.2%} is unrealistic"
    
    def test_no_nulls_in_critical_columns(self, train_data):
        """Critical columns must not have null values."""
        critical = ["amount", "hour", "day_of_week", "merchant_category", "is_fraud"]
        for col in critical:
            null_count = train_data[col].isnull().sum()
            assert null_count == 0, f"Column {col} has {null_count} null values"


class TestModelPerformance:
    """Tests for model performance thresholds."""
    
    @pytest.fixture
    def test_data(self):
        if not os.path.exists("data/test.csv"):
            pytest.skip("data/test.csv not found")
        return pd.read_csv("data/test.csv")
    
    @pytest.fixture
    def model_package(self):
        if not os.path.exists("models/model.pkl"):
            pytest.skip("models/model.pkl not found")
        with open("models/model.pkl", "rb") as f:
            package = pickle.load(f)
            # Handle both old and new formats
            if isinstance(package, tuple) and len(package) == 3:
                return package
            elif isinstance(package, tuple) and len(package) == 2:
                return package[0], package[1], None
            return None

    def test_model_loads_successfully(self, model_package):
        """Model file must load without errors."""
        assert model_package is not None, "Model package is None"
        model, encoder, _ = model_package
        assert model is not None, "Model is None"
        assert encoder is not None, "Encoder is None"
    
    def test_model_can_predict(self, model_package, test_data):
        """Model must be able to make predictions."""
        model, encoder, _ = model_package
        test_data_copy = test_data.copy()
        test_data_copy["merchant_encoded"] = encoder.transform(test_data_copy["merchant_category"])
        X = test_data_copy[["amount", "hour", "day_of_week", "merchant_encoded"]]
        predictions = model.predict(X)
        assert len(predictions) == len(X), "Prediction count mismatch"
    
    def test_accuracy_threshold(self, model_package, test_data):
        """Model accuracy must be at least 90%."""
        model, encoder, _ = model_package
        test_data_copy = test_data.copy()
        test_data_copy["merchant_encoded"] = encoder.transform(test_data_copy["merchant_category"])
        X = test_data_copy[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data_copy["is_fraud"]
        accuracy = model.score(X, y)
        assert accuracy >= 0.90, f"Accuracy {accuracy:.2%} below 90% threshold"
    
    def test_f1_threshold(self, model_package, test_data):
        """Model F1-score must be at least 0.3."""
        model, encoder, _ = model_package
        test_data_copy = test_data.copy()
        test_data_copy["merchant_encoded"] = encoder.transform(test_data_copy["merchant_category"])
        X = test_data_copy[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data_copy["is_fraud"]
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred)
        assert f1 >= 0.3, f"F1-score {f1:.2f} below 0.3 threshold"
    
    def test_precision_not_zero(self, model_package, test_data):
        """Model precision must be greater than 0."""
        model, encoder, _ = model_package
        test_data_copy = test_data.copy()
        test_data_copy["merchant_encoded"] = encoder.transform(test_data_copy["merchant_category"])
        X = test_data_copy[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data_copy["is_fraud"]
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred, zero_division=0)
        assert precision > 0, "Model has zero precision"
    
    def test_recall_not_zero(self, model_package, test_data):
        """Model recall must be greater than 0."""
        model, encoder, _ = model_package
        test_data_copy = test_data.copy()
        test_data_copy["merchant_encoded"] = encoder.transform(test_data_copy["merchant_category"])
        X = test_data_copy[["amount", "hour", "day_of_week", "merchant_encoded"]]
        y = test_data_copy["is_fraud"]
        y_pred = model.predict(X)
        recall = recall_score(y, y_pred, zero_division=0)
        assert recall > 0, "Model has zero recall"
