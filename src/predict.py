import joblib
import numpy as np
import pandas as pd
from .preprocess import create_clinical_features

class DiabetesPredictor:
    def __init__(self, model_path='models/diabetes_random_forest.pkl'):
        """
        Initialize diabetes predictor
        
        Args:
            model_path (str): Path to trained model
        """
        self.model = joblib.load(model_path)
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
    
    def predict(self, features, return_prob=True):
        """
        Predict diabetes risk
        
        Args:
            features (list/dict): Patient data
            return_prob (bool): Return probability or class
            
        Returns:
            float or int: Diabetes risk (probability or class)
        """
        # Convert input to proper format
        if isinstance(features, dict):
            input_data = pd.DataFrame([features])
        elif isinstance(features, list):
            input_data = pd.DataFrame([features], columns=self.feature_names)
        else:
            raise ValueError("Input must be list or dictionary")
        
        # Add clinical features
        input_data = create_clinical_features(input_data)
        
        # Ensure correct feature order
        input_data = input_data[self.model.feature_names_in_]
        
        # Make prediction
        if return_prob:
            return self.model.predict_proba(input_data)[0][1]
        return self.model.predict(input_data)[0]
    
    def explain_prediction(self, features):
        """
        Generate explanation for prediction (SHAP values)
        
        Requires: pip install shap
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return None
        
        # Prepare input
        input_data = pd.DataFrame([features], columns=self.feature_names)
        input_data = create_clinical_features(input_data)
        input_data = input_data[self.model.feature_names_in_]
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(input_data)
        
        # Return explanation
        return {
            'base_value': explainer.expected_value[1],
            'shap_values': dict(zip(input_data.columns, shap_values[1][0])),
            'prediction': self.model.predict_proba(input_data)[0][1]
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Sample patient data
    patient = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 100,
        'BMI': 26.2,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 35
    }
    
    # Get prediction
    risk = predictor.predict(patient)
    print(f"Diabetes risk: {risk:.1%}")
    
    # Get explanation
    explanation = predictor.explain_prediction(patient)
    if explanation:
        print("\nPrediction Explanation:")
        for feature, value in explanation['shap_values'].items():
            print(f"{feature}: {value:.4f}")
        print(f"Base value: {explanation['base_value']:.4f}")
        print(f"Final prediction: {explanation['prediction']:.4f}")