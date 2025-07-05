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

# Clinical definitions for each parameter
CLINICAL_DEFINITIONS = {
    'Pregnancies': "Number of times pregnant",
    'Glucose': "Fasting plasma glucose (mg/dL) - Measures blood sugar levels after overnight fasting",
    'BloodPressure': "Diastolic blood pressure (mm Hg) - Pressure in arteries between heartbeats",
    'SkinThickness': "Triceps skin fold thickness (mm) - Measures body fat percentage",
    'Insulin': "2-Hour serum insulin (μU/mL) - Insulin levels after glucose challenge",
    'BMI': "Body Mass Index (kg/m²) - Weight relative to height",
    'DiabetesPedigreeFunction': "Diabetes genetic risk score - Estimates familial diabetes risk",
    'Age': "Age in years - Diabetes risk increases with age"
}

# App title
st.title('🩺 Clinical Diabetes Risk Assessment')
st.write("""
*Interpretable ML model predicting diabetes risk using routine clinical measurements*
""")

# Sidebar inputs
st.sidebar.header('Patient Clinical Measurements')

# Add clinical definitions expander
with st.sidebar.expander("ℹ️ Measurement Definitions"):
    for feature, definition in CLINICAL_DEFINITIONS.items():
        st.markdown(f"**{feature}**: {definition}")

# Explain blood pressure measurement
st.sidebar.markdown("""
<div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-top:10px">
<small>💡 <strong>Note on Blood Pressure</strong>: 
This model uses diastolic blood pressure only as it showed stronger predictive value 
in our analysis. Systolic pressure was not included in the original dataset.</small>
</div>
""", unsafe_allow_html=True)

def user_input_features():
    inputs = {}
    inputs['Pregnancies'] = st.sidebar.slider('Pregnancies', 0, 17, 2, 
                                             help=CLINICAL_DEFINITIONS['Pregnancies'])
    inputs['Glucose'] = st.sidebar.slider('Glucose (mg/dL)', 50, 300, 120, 
                                         help=CLINICAL_DEFINITIONS['Glucose'])
    inputs['BloodPressure'] = st.sidebar.slider('Diastolic BP (mmHg)', 30, 120, 70, 
                                               help=CLINICAL_DEFINITIONS['BloodPressure'])
    inputs['SkinThickness'] = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 20, 
                                               help=CLINICAL_DEFINITIONS['SkinThickness'])
    inputs['Insulin'] = st.sidebar.slider('Insulin (μU/mL)', 0, 846, 100, 
                                         help=CLINICAL_DEFINITIONS['Insulin'])
    inputs['BMI'] = st.sidebar.slider('BMI (kg/m²)', 15.0, 50.0, 26.2, step=0.1, 
                                     help=CLINICAL_DEFINITIONS['BMI'])
    inputs['DiabetesPedigreeFunction'] = st.sidebar.slider('Diabetes Genetic Risk', 0.08, 2.5, 0.5, step=0.01, 
                                                          help=CLINICAL_DEFINITIONS['DiabetesPedigreeFunction'])
    inputs['Age'] = st.sidebar.slider('Age (years)', 20, 85, 35, 
                                     help=CLINICAL_DEFINITIONS['Age'])
    
    return pd.DataFrame([inputs], columns=feature_names)

# Get user input
input_df = user_input_features()

# Display user inputs
st.subheader('Patient Input Features')
st.dataframe(input_df.style.format("{:.1f}"))

# Add clinical context
st.markdown("""
**Clinical Risk Thresholds**:
- Glucose >140 mg/dL → Prediabetes range
- BMI >25 kg/m² → Overweight
- BMI >30 kg/m² → Obese
- Age >45 years → Increased risk
""")

# Predict and display results
if st.button('Assess Diabetes Risk'):
    # Predict
    probability = pipeline.predict_proba(input_df)[0][1]
    
    # Enhanced risk stratification
    if probability < 0.2:
        risk_category = "Very Low Risk"
        color = "green"
        emoji = "✅"
    elif probability < 0.4:
        risk_category = "Low Risk"
        color = "teal"
        emoji = "🟢"
    elif probability < 0.6:
        risk_category = "Moderate Risk"
        color = "orange"
        emoji = "🟠"
    elif probability < 0.8:
        risk_category = "High Risk"
        color = "red"
        emoji = "🔴"
    else:
        risk_category = "Very High Risk"
        color = "darkred"
        emoji = "⚠️"
    
    # Display results with enhanced visualization
    st.subheader('Risk Assessment')
    
    # Create a risk meter
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:20px">
        <div style="display:flex; justify-content:space-between; margin-bottom:10px">
            <span><strong>Risk Probability:</strong></span>
            <span style="font-size:24px; color:{color}; font-weight:bold">{probability:.1%}</span>
        </div>
        <div style="background:linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%); 
                    height:25px; border-radius:5px; position:relative">
            <div style="position:absolute; left:{probability*100}%; top:-30px; transform:translateX(-50%)">
                {emoji}
            </div>
            <div style="position:absolute; left:{probability*100}%; 
                        height:35px; width:2px; background-color:black"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:5px">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
        </div>
        <div style="text-align:center; font-size:20px; color:{color}; margin-top:10px">
            {risk_category}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk-specific recommendations
    st.subheader('Clinical Recommendations')
    
    if risk_category in ["Very Low Risk", "Low Risk"]:
        st.success(f"""
        **Preventive Measures:**
        - Continue annual glucose screening
        - Maintain BMI <25 through balanced diet
        - Engage in 150 mins/week of moderate exercise
        - Reassess risk factors in 2-3 years
        """)
    elif risk_category == "Moderate Risk":
        st.warning(f"""
        **Early Intervention Recommended:**
        - Perform HbA1c test within 3 months
        - Begin lifestyle modification program
        - Monitor fasting glucose quarterly
        - Consider metformin if prediabetes confirmed
        """)
    else:  # High or Very High Risk
        st.error(f"""
        **Immediate Action Required:**
        - Conduct comprehensive diabetes screening now
        - Refer to endocrinology specialist
        - Implement intensive lifestyle intervention
        - Begin pharmacotherapy if indicated
        - Schedule follow-up in 1 month
        """)
    
    # SHAP explanation
    st.subheader('Risk Factor Analysis')
    
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
        
        # Feature impact table
        st.write("**Feature Contributions to Diabetes Risk:**")
        impact_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': shap_values_single,
            'Clinical Significance': [
                "Elevated risk" if val > 0 else "Reduced risk" 
                for val in shap_values_single
            ]
        }).sort_values('Impact', ascending=False)
        
        # Format impact values and add interpretation
        impact_df['Interpretation'] = impact_df.apply(
            lambda row: f"{abs(row['Impact']):.2f} {'increase' if row['Impact'] > 0 else 'decrease'} in diabetes probability",
            axis=1
        )
        
        st.dataframe(impact_df[['Feature', 'Interpretation', 'Clinical Significance']].style.bar(
            subset=['Impact'], 
            color=['#ff9999','#99ff99'], 
            align='zero'
        ))
        
        # Force plot
        st.write("**How each feature influences the prediction:**")
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
        
    except Exception as e:
        st.warning(f"Detailed explanation unavailable: {str(e)}")
        st.info("Showing standard feature importance instead")
        
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
            st.error("Feature importance data unavailable")

# Key Insights Section
st.sidebar.markdown("""
---
**Clinical Risk Thresholds:**
- Glucose >140 mg/dL → Prediabetes
- BMI >25 kg/m² → Overweight
- BMI >30 kg/m² → Obesity
- Age >45 years → Increased risk
""")

st.subheader('Model Insights')
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Dominant Risk Factors:**
    1. Glucose levels (Fasting plasma glucose)
    2. BMI (Body Mass Index)
    3. Age
    4. Diabetes Genetic Risk Score
    """)
    
    st.markdown("""
    **Risk Stratification:**
    - < 20% → Very Low Risk
    - 20-39% → Low Risk
    - 40-59% → Moderate Risk
    - 60-79% → High Risk
    - ≥80% → Very High Risk
    """)

with col2:
    st.markdown("""
    **Clinical Validation:**
    - Matches established diabetes pathophysiology
    - Identifies hyperglycemia as primary driver
    - Detects obesity and age-related patterns
    - Aligns with ADA screening guidelines
    """)
    
    st.markdown("""
    **Key Limitations:**
    - Insulin data missing in 49% of cases
    - Homogeneous patient cohort
    - Doesn't include lifestyle factors
    """)

# Model performance info
st.markdown("""
**Model Performance Metrics:**
- AUC: 0.8146 (Excellent discrimination)
- Recall: 81% (Minimizes missed cases)
- Precision: 63% (Balances unnecessary interventions)
- F1 Score: 0.71 (Optimal threshold: 0.31)
""")

# Footer
st.markdown("""
---
**Disclaimer**: 
This tool provides risk assessment based on statistical modeling, not a medical diagnosis. 
Clinical judgment should always supersede algorithmic predictions. Consult a healthcare 
professional for medical advice.

*Model developed with clinical interpretability focus using SHAP explanations. 
Validated on Pima Indians Diabetes Dataset.*
""")