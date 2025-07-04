import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from .preprocess import load_and_clean

def train_model(data_path, model_type='random_forest', save_path='models/'):
    """
    Train and save diabetes prediction model
    
    Args:
        data_path (str): Path to diabetes.csv
        model_type (str): 'random_forest' or 'xgboost'
        save_path (str): Directory to save model
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_clean(data_path)
    
    # Model selection
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            eval_metric='logloss',
            random_state=42,
        )
    else:
        raise ValueError("Invalid model_type. Choose 'random_forest' or 'xgboost'")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, f"{save_path}/diabetes_{model_type}.pkl")
    print(f"Model trained and saved to {save_path}/diabetes_{model_type}.pkl")
    
    # Return test set for evaluation
    return model, X_test, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train diabetes prediction model')
    parser.add_argument('--data', type=str, default='data/diabetes.csv',
                        help='Path to diabetes dataset')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost'],
                        help='Model type to train')
    parser.add_argument('--output', type=str, default='models/',
                        help='Output directory for model')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_type=args.model,
        save_path=args.output
    )