import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Load model and features
@st.cache_resource
def load_model():
    pipeline = joblib.load('clinical_diabetes_pipeline.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return pipeline, feature_names

pipeline, feature_names = load_model()

# App title
st.title('🩺 Clinical Diabetes Risk Assessment')
st.write("""
*Interpretable ML model predicting diabetes risk using clinical measurements*
""")

# Sidebar inputs
st.sidebar.header('Patient Clinical Measurements')

def user_input_features():
    inputs = {}
    inputs['Pregnancies'] = st.sidebar.slider('Pregnancies', 0, 17, 2)
    inputs['Glucose'] = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120)
    inputs['BloodPressure'] = st.sidebar.slider('Blood Pressure (mmHg)', 0, 122, 70)
    inputs['SkinThickness'] = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 20)
    inputs['Insulin'] = st.sidebar.slider('Insulin (μU/mL)', 0, 846, 100)
    inputs['BMI'] = st.sidebar.slider('BMI (kg/m²)', 0.0, 67.1, 26.2)
    inputs['DiabetesPedigreeFunction'] = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.5)
    inputs['Age'] = st.sidebar.slider('Age (years)', 21, 81, 35)
    
    return pd.DataFrame([inputs], columns=feature_names)

# Get user input
input_df = user_input_features()

# Display user inputs
st.subheader('Patient Input Features')
st.dataframe(input_df)

# Predict and display results
if st.button('Assess Diabetes Risk'):
    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    
    # Display results
    st.subheader('Risk Assessment')
    if prediction == 1:
        st.error(f'High risk of diabetes (probability: {probability:.1%})')
        st.write("""
        **Clinical Recommendations:**
        - Recommend HbA1c test
        - Lifestyle intervention program
        - 3-month follow-up
        """)
    else:
        st.success(f'Low risk of diabetes (probability: {probability:.1%})')
        st.write("""
        **Clinical Recommendations:**
        - Annual glucose screening
        - Maintain healthy BMI
        - Regular exercise
        """)
    
    # SHAP explanation
    st.subheader('Risk Factor Explanation')
    
    try:
        # Extract model from pipeline
        model = pipeline.named_steps['classifier']
        preprocessor = Pipeline(pipeline.steps[:-1])
        transformed_input = preprocessor.transform(input_df)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values_calc = explainer.shap_values(transformed_input)
        
        # Handle different SHAP output formats
        if isinstance(shap_values_calc, list) and len(shap_values_calc) == 2:
            shap_values_positive = np.array(shap_values_calc[1])
            base_value = explainer.expected_value[1]
        elif len(np.array(shap_values_calc).shape) == 3:
            shap_values_positive = shap_values_calc[:, :, 1]
            base_value = explainer.expected_value[1]
        else:
            shap_values_positive = np.array(shap_values_calc)
            base_value = explainer.expected_value

        shap_values_single = shap_values_positive[0]
        
        # Force plot
        st.write("**How each feature impacts the prediction:**")
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            base_value,
            shap_values_single,
            input_df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Feature impact table
        st.write("**Feature Contributions:**")
        impact_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_values_single
        }).sort_values('Impact', ascending=False)
        
        st.dataframe(impact_df.style.bar(color=['#ff9999','#99ff99'], align='zero'))
        
    except Exception as e:
        st.warning(f"Couldn't generate detailed explanation: {str(e)}")
        st.info("Showing simple feature importance instead")
        
        # Fallback to standard feature importance
        try:
            model = pipeline.named_steps['classifier']
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))
        except:
            st.error("Couldn't retrieve feature importance")

# Key Insights Section
st.sidebar.markdown("""
---
**Clinical Risk Thresholds:**
- Glucose >140 mg/dL
- BMI >25 kg/m²
- Age >40 years
""")

st.subheader('Model Insights')
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Dominant Risk Factors:**
    1. Glucose levels
    2. BMI
    3. Age
    4. Diabetes Pedigree Function
    """)

with col2:
    st.markdown("""
    **Clinical Validation:**
    - Matches known pathophysiology
    - Identifies hyperglycemia as primary driver
    - Detects obesity and age patterns
    """)

# Model performance info
st.markdown("""
**Model Performance:**
- AUC: 0.8146
- Recall: 81% (optimized threshold)
- Precision: 63%
""")

# Footer
st.markdown("""
---
*Model developed with clinical interpretability focus using SHAP explanations. Not for diagnostic use.*
""")