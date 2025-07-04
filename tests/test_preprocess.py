import pytest
import pandas as pd
import numpy as np
from src.preprocess import load_and_clean, create_clinical_features

def test_load_and_clean_handles_zeros(sample_data, tmp_path):
    # Save sample data to a temp CSV
    data_path = tmp_path / "diabetes.csv"
    sample_data.to_csv(data_path, index=False)
    

    # Test loading and cleaning
    X_train, X_test, y_train, y_test, feature_names = load_and_clean(data_path)
    
    # Check if zeros in specified columns are replaced
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
    
    # Check that insulin zero was handled
    assert X_train.loc[X_train['Insulin'] == 0, 'Insulin'].empty

def test_create_clinical_features(sample_data):
    df = create_clinical_features(sample_data)
    
    # Check new features are created
    assert 'Glucose_BMI' in df.columns
    assert 'Age_Pregnancies' in df.columns
    assert 'BP_Glucose_Ratio' in df.columns
    assert 'Metabolic_Index' in df.columns
    
    # Check calculations
    assert df.loc[0, 'Glucose_BMI'] == pytest.approx(120 * 26.2)
    assert df.loc[0, 'Age_Pregnancies'] == pytest.approx(35 * 2)
    assert df.loc[0, 'BP_Glucose_Ratio'] == pytest.approx(70 / 120)
    assert df.loc[0, 'Metabolic_Index'] == pytest.approx((120 * 100) / 100)
    
    # Check no inf or nan values
    assert not df.isin([np.inf, -np.inf]).any().any()
    assert not df.isna().any().any()

def test_data_split_balance(sample_data, tmp_path):
    data_path = tmp_path / "diabetes.csv"
    sample_data.to_csv(data_path, index=False)
    
    X_train, X_test, y_train, y_test, _ = load_and_clean(data_path)
    
    # Check class distribution in splits
    train_balance = y_train.value_counts(normalize=True)
    test_balance = y_test.value_counts(normalize=True)
    original_balance = sample_data['Outcome'].value_counts(normalize=True)
    
    assert pytest.approx(train_balance[0], abs=0.1) == original_balance[0]
    assert pytest.approx(test_balance[0], abs=0.1) == original_balance[0]