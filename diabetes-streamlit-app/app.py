import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import os  # Added for path handling

# Load model and features with robust path handling
@st.cache_resource
def load_model():
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths to model files
    pipeline_path = os.path.join(current_dir, 'clinical_diabetes_pipeline.pkl')
    features_path = os.path.join(current_dir, 'feature_names.pkl')
    
    # Load with explicit path verification
    if not os.path.exists(pipeline_path):
        st.error(f"Model file not found at: {pipeline_path}")
    if not os.path.exists(features_path):
        st.error(f"Feature names file not found at: {features_path}")
    
    pipeline = joblib.load(pipeline_path)
    feature_names = joblib.load(features_path)
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

# Add clinical risk thresholds
st.markdown("""
**Clinical Risk Thresholds**:
- 🟢 Glucose < 100 mg/dL (Normal)
- 🟡 Glucose 100-125 mg/dL (Prediabetes)
- 🔴 Glucose ≥ 126 mg/dL (Diabetes)
- 🟢 BMI < 25 kg/m² (Healthy)
- 🟡 BMI 25-30 kg/m² (Overweight)
- 🔴 BMI > 30 kg/m² (Obese)
- 🔺 Age > 45 years (Increased risk)
""")

# Predict and display results
if st.button('Assess Diabetes Risk'):
    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    
    # Enhanced risk stratification
    if probability < 0.2:
        risk_category = "Very Low Risk"
        color = "#4CAF50"  # Green
        emoji = "✅"
        risk_level = 1
    elif probability < 0.4:
        risk_category = "Low Risk"
        color = "#8BC34A"  # Light green
        emoji = "🟢"
        risk_level = 2
    elif probability < 0.6:
        risk_category = "Moderate Risk"
        color = "#FFC107"  # Amber
        emoji = "🟠"
        risk_level = 3
    elif probability < 0.8:
        risk_category = "High Risk"
        color = "#FF9800"  # Orange
        emoji = "🔴"
        risk_level = 4
    else:
        risk_category = "Very High Risk"
        color = "#F44336"  # Red
        emoji = "⚠️"
        risk_level = 5
    
    # Display results with enhanced visualization
    st.subheader('Diabetes Risk Assessment')
    
    # Create risk card
    st.markdown(f"""
    <div style="border-radius:10px; padding:20px; background-color:#f8f9fa; 
                border-left: 6px solid {color}; margin-bottom:20px">
        <div style="display:flex; align-items:center; gap:20px">
            <div style="font-size:48px; color:{color}">{emoji}</div>
            <div>
                <h2 style="margin:0; color:{color}">{risk_category}</h2>
                <p style="font-size:24px; margin:10px 0; font-weight:bold; color:{color}">
                    {probability:.1%} probability
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create simple risk meter
    st.write("**Risk Level:**")
    fig, ax = plt.subplots(figsize=(10, 2))
    risk_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    risk_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
    
    # Find current risk level index
    risk_level_idx = 0
    if probability >= 0.8: risk_level_idx = 4
    elif probability >= 0.6: risk_level_idx = 3
    elif probability >= 0.4: risk_level_idx = 2
    elif probability >= 0.2: risk_level_idx = 1
    
    for i in range(5):
        ax.barh(0, 20, left=i*20, color=risk_colors[i], alpha=0.7 if i != risk_level_idx else 1.0)
        ax.text(i*20 + 10, 0, risk_labels[i], 
                ha='center', va='center', color='white' if i == risk_level_idx else 'black')
    
    # Add indicator
    ax.plot([probability*100, probability*100], [-0.5, 0.5], 'k-', linewidth=2)
    ax.text(probability*100, -0.7, f'{probability:.1%}', ha='center')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    st.pyplot(fig)
    
    # Risk-specific recommendations
    st.subheader('Clinical Recommendations')
    
    if risk_level <= 2:  # Very Low or Low Risk
        st.success(f"""
        **Preventive Measures:**
        - Continue annual glucose screening
        - Maintain BMI <25 through balanced diet
        - Engage in 150 mins/week of moderate exercise
        - Reassess risk factors in 2-3 years
        """)
    elif risk_level == 3:  # Moderate Risk
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
    
    # Feature importance explanation
    st.subheader('Key Risk Factors')
    
    try:
        # Get feature importances
        model = pipeline.named_steps['classifier']
        importance = model.feature_importances_
        
        # Create impact DataFrame
        impact_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Display top factors
        st.write("**Most Important Risk Factors:**")
        top_factors = impact_df.head(3)
        cols = st.columns(3)
        
        for i, (_, row) in enumerate(top_factors.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; text-align:center; 
                            border-left: 4px solid #ff9999; margin-bottom:15px">
                    <div style="font-weight:bold; font-size:16px">{row['Feature']}</div>
                    <div style="font-size:24px; color:#ff9999; margin:10px 0">▲</div>
                    <div style="font-size:14px">Impact: {row['Importance']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Display all factors
        st.write("**All Risk Factors:**")
        st.dataframe(impact_df, hide_index=True)
        
        # Create horizontal bar chart
        st.subheader("Risk Factor Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        impact_df = impact_df.sort_values('Importance', ascending=True)
        
        bars = ax.barh(impact_df['Feature'], impact_df['Importance'], color='#4CAF50')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Feature importance data unavailable: {str(e)}")

# Key Insights Section
st.sidebar.markdown("""
---
**Clinical Risk Thresholds**:
- Glucose >140 mg/dL → Prediabetes
- BMI >25 kg/m² → Overweight
- BMI >30 kg/m² → Obesity
- Age >45 years → Increased risk
""")

st.subheader('Model Insights')
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Dominant Risk Factors**:
    1. Glucose levels
    2. BMI
    3. Age
    4. Diabetes Genetic Risk
    """)
    
    st.markdown("""
    **Risk Stratification**:
    - < 20% → Very Low Risk
    - 20-39% → Low Risk
    - 40-59% → Moderate Risk
    - 60-79% → High Risk
    - ≥80% → Very High Risk
    """)

with col2:
    st.markdown("""
    **Clinical Validation**:
    - Matches established diabetes pathophysiology
    - Identifies hyperglycemia as primary driver
    - Detects obesity and age-related patterns
    """)
    
    st.markdown("""
    **Key Limitations**:
    - Insulin data missing in 49% of cases
    - Homogeneous patient cohort
    - Doesn't include lifestyle factors
    """)

# Model performance info
st.markdown("""
**Model Performance Metrics**:
- AUC: 0.8146
- Recall: 81%
- Precision: 63%
- F1 Score: 0.71
""")

# Footer
st.markdown("""
---
**Disclaimer**: 
This tool provides risk assessment based on statistical modeling, not a medical diagnosis. 
Clinical judgment should always supersede algorithmic predictions. 

*Model developed for clinical interpretability. Validated on Pima Indians Diabetes Dataset.*
""")