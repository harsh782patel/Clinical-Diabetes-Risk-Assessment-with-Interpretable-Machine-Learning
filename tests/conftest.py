import pytest
import pandas as pd
import matplotlib
import numpy as np
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    """Sample diabetes dataset for testing - 20 samples"""
    data = {
        'Pregnancies': [2, 5, 1, 0, 3, 4, 2, 1, 0, 3, 1, 6, 2, 0, 4, 3, 2, 1, 0, 2],
        'Glucose': [120, 180, 98, 142, 92, 200, 130, 105, 90, 110, 
                    85, 150, 125, 135, 115, 160, 95, 140, 155, 100],
        'BloodPressure': [70, 82, 76, 64, 90, 85, 75, 65, 70, 80,
                          60, 72, 68, 74, 78, 66, 62, 84, 70, 76],
        'SkinThickness': [20, 35, 15, 32, 28, 40, 25, 30, 22, 35,
                          18, 38, 29, 31, 33, 27, 24, 36, 26, 30],
        'Insulin': [100, 0, 80, 150, 120, 0, 100, 140, 90, 120,
                    85, 200, 110, 130, 95, 180, 70, 160, 145, 115],
        'BMI': [26.2, 33.1, 23.1, 35.3, 28.7, 31.2, 27.5, 29.0, 25.5, 30.1,
                22.8, 34.5, 28.2, 32.8, 29.5, 31.8, 24.6, 33.5, 30.2, 27.9],
        'DiabetesPedigreeFunction': [0.5, 0.8, 0.2, 0.9, 1.2, 0.7, 0.6, 0.3, 0.4, 1.0,
                                     0.35, 0.85, 0.55, 0.75, 0.65, 0.95, 0.25, 0.8, 0.45, 0.6],
        'Age': [35, 42, 28, 30, 45, 50, 32, 29, 31, 40,
                26, 48, 33, 36, 39, 52, 27, 44, 38, 34],
        'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_train_data():
    """Generated classification data for model training tests"""
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    return pd.DataFrame(X, columns=feature_names), y

matplotlib.use('Agg')

@pytest.fixture(autouse=True)
def set_matplotlib_backend():
    # Ensure non-interactive backend for tests
    matplotlib.use('Agg')

@pytest.fixture
def trained_model(sample_train_data):
    """Simple trained model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    X, y = sample_train_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model