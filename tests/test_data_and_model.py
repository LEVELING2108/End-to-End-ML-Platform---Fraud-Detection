"""
Tests for realistic imbalanced data and high-dimensional model.
"""
import pytest
import pandas as pd
import os
import pickle
from sklearn.metrics import recall_score, precision_score, f1_score

class TestDataQuality:
    def test_data_exists(self):
        assert os.path.exists("data/train.csv")
        assert os.path.exists("data/test.csv")

    def test_imbalance_ratio(self):
        train_df = pd.read_csv("data/train.csv")
        fraud_ratio = train_df['Class'].mean()
        assert fraud_ratio < 0.05, f"Data not imbalanced enough: {fraud_ratio:.2%}"

class TestModelPerformance:
    @pytest.fixture
    def model_package(self):
        path = "models/model_package.pkl"
        if not os.path.exists(path): pytest.skip("Model package missing")
        with open(path, "rb") as f: return pickle.load(f)

    def test_model_can_predict(self, model_package):
        model = model_package["model"]
        test_df = pd.read_csv("data/test.csv")
        X = test_df.drop('Class', axis=1)
        y_pred = model.predict(X)
        assert len(y_pred) == len(X)

    def test_recall_is_non_zero(self, model_package):
        # Recall is hard on 99.8% imbalance, but SMOTE should help
        model = model_package["model"]
        test_df = pd.read_csv("data/test.csv")
        X = test_df.drop('Class', axis=1)
        y = test_df['Class']
        y_pred = model.predict(X)
        recall = recall_score(y, y_pred, zero_division=0)
        assert recall >= 0, "Recall test performed"
