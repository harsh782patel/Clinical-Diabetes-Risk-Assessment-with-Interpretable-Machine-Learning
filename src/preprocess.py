import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean(file_path):
    """
    Load and preprocess diabetes dataset
    
    Args:
        file_path (str): Path to diabetes.csv file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Load raw data
    df = pd.read_csv(file_path)
    
    # Handle biological zeros
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_features] = df[zero_features].replace(0, np.nan)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Separate features and target
    X = df_imputed.drop('Outcome', axis=1)
    y = df_imputed['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def create_clinical_features(df):
    """
    Create clinically relevant feature interactions
    
    Args:
        df (DataFrame): Input data
        
    Returns:
        DataFrame: Data with new features
    """
    # Clinical feature engineering
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Pregnancies'] = df['Age'] * df['Pregnancies']
    df['BP_Glucose_Ratio'] = df['BloodPressure'] / df['Glucose']
    df['Metabolic_Index'] = (df['Glucose'] * df['Insulin']) / 100
    
    # Handle potential infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    return df