import pytest
from src.train import train_model
import joblib
import os

def test_train_model_creates_file(sample_data, tmp_path):
    # Prepare test data
    data_path = tmp_path / "diabetes.csv"
    sample_data.to_csv(data_path, index=False)
    model_path = tmp_path / "models"
    model_path.mkdir()
    
    # Train model
    train_model(data_path=str(data_path), model_type='random_forest', save_path=str(model_path))
    
    # Check model file was created
    files = os.listdir(model_path)
    assert 'diabetes_random_forest.pkl' in files
    
    # Load model and verify
    model = joblib.load(model_path / 'diabetes_random_forest.pkl')
    assert hasattr(model, 'predict_proba')

def test_train_model_xgboost(sample_data, tmp_path):
    # Prepare test data
    data_path = tmp_path / "diabetes.csv"
    sample_data.to_csv(data_path, index=False)
    model_path = tmp_path / "models"
    model_path.mkdir()
    
    # Train model
    train_model(data_path=str(data_path), model_type='xgboost', save_path=str(model_path))
    
    # Check model file was created
    assert 'diabetes_xgboost.pkl' in os.listdir(model_path)

def test_train_model_invalid_type(sample_data, tmp_path):
    data_path = tmp_path / "diabetes.csv"
    sample_data.to_csv(data_path, index=False)
    
    with pytest.raises(ValueError):
        train_model(data_path=str(data_path), model_type='invalid_model')