import pytest
import numpy as np
from src.evaluate import evaluate_model, compare_models
from sklearn.metrics import roc_auc_score
import os
def test_evaluate_model_output(trained_model, sample_train_data, tmp_path):
    X, y = sample_train_data
    X_test = X.iloc[:20]
    y_test = y[:20]
    
    # Capture printed output
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        results = evaluate_model(trained_model, X_test, y_test, "Test Model", save_path=str(tmp_path))
    
    output = f.getvalue()
    
    # Check printed metrics
    assert "Test Model Performance:" in output
    assert "ROC AUC:" in output
    assert "precision" in output
    
    # Check returned results
    assert 'classification_report' in results
    assert 'roc_auc' in results
    assert isinstance(results['roc_auc'], float)
    assert 0 <= results['roc_auc'] <= 1

def test_compare_models_visualization(trained_model, sample_train_data, tmp_path):
    X, y = sample_train_data
    X_test = X.iloc[:20]
    y_test = y[:20]
    
    # Create multiple model results
    results_dict = {
        'Model1': evaluate_model(trained_model, X_test, y_test, "Model1"),
        'Model2': evaluate_model(trained_model, X_test, y_test, "Model2")
    }
    
    # Test comparison
    compare_models(results_dict, save_path=str(tmp_path))
    
    # Check visualization was created
    assert 'model_comparison.png' in os.listdir(tmp_path)

def test_evaluate_model_feature_importance(trained_model, sample_train_data):
    X, y = sample_train_data
    results = evaluate_model(trained_model, X, y, "Test Model")
    
    # Check feature importance was handled
    # (No direct output check, just ensure no errors)
    assert True