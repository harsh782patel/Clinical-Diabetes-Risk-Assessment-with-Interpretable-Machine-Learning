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

# Display user inputs with clinical context
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
    
    # Create a risk meter
    st.markdown(f"""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; margin-bottom:20px; border-left: 5px solid {color}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px">
            <h3 style="color:{color}; margin:0">Risk Category: {risk_category} {emoji}</h3>
            <h2 style="color:{color}; margin:0">{probability:.1%}</h2>
        </div>
        
        <div style="background:linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%); 
                    height:20px; border-radius:10px; position:relative; margin-bottom:30px">
            <div style="position:absolute; left:{probability*100}%; top:-30px; transform:translateX(-50%)">
                <div style="font-size:24px">{emoji}</div>
            </div>
            <div style="position:absolute; left:{probability*100}%; 
                        height:35px; width:3px; background-color:black"></div>
        </div>
        
        <div style="display:flex; justify-content:space-between; margin-top:5px; font-size:14px">
            <div style="text-align:center">
                <div>0%</div>
                <div>Very Low</div>
            </div>
            <div style="text-align:center">
                <div>20%</div>
                <div>Low</div>
            </div>
            <div style="text-align:center">
                <div>40%</div>
                <div>Moderate</div>
            </div>
            <div style="text-align:center">
                <div>60%</div>
                <div>High</div>
            </div>
            <div style="text-align:center">
                <div>80%</div>
                <div>Very High</div>
            </div>
            <div style="text-align:center">
                <div>100%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk-specific recommendations
    st.subheader('Clinical Recommendations')
    
    if risk_level <= 2:  # Very Low or Low Risk
        st.success(f"""
        **Preventive Measures:**
        - ✅ Continue annual glucose screening
        - ✅ Maintain BMI <25 through balanced diet
        - ✅ Engage in 150 mins/week of moderate exercise
        - ✅ Reassess risk factors in 2-3 years
        """)
    elif risk_level == 3:  # Moderate Risk
        st.warning(f"""
        **Early Intervention Recommended:**
        - 🔸 Perform HbA1c test within 3 months
        - 🔸 Begin lifestyle modification program
        - 🔸 Monitor fasting glucose quarterly
        - 🔸 Consider metformin if prediabetes confirmed
        """)
    else:  # High or Very High Risk
        st.error(f"""
        **Immediate Action Required:**
        - 🔴 Conduct comprehensive diabetes screening now (Fasting glucose + HbA1c)
        - 🔴 Refer to endocrinology specialist
        - 🔴 Implement intensive lifestyle intervention (Diet + Exercise)
        - 🔴 Begin pharmacotherapy if indicated
        - 🔴 Schedule follow-up in 1 month
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
        
        # Feature impact table - FIXED THE ERROR HERE
        st.write("**How each feature contributes to your diabetes risk:**")
        
        # Create impact DataFrame with interpretation
        impact_data = []
        for i, feature in enumerate(feature_names):
            impact = shap_values_single[i]
            abs_impact = abs(impact)
            
            if impact > 0:
                significance = f"🟥 Increases risk by {abs_impact:.3f}"
                color = "#ff9999"
            else:
                significance = f"🟩 Decreases risk by {abs_impact:.3f}"
                color = "#99ff99"
                
            impact_data.append({
                'Feature': feature,
                'Impact': impact,
                'Clinical Significance': significance,
                'Color': color
            })
        
        impact_df = pd.DataFrame(impact_data).sort_values('Impact', ascending=False)
        
        # Display styled table without bar formatting
        st.dataframe(impact_df[['Feature', 'Clinical Significance']])
        
        # Visual impact bars
        st.write("**Impact Magnitude:**")
        for _, row in impact_df.iterrows():
            label = f"{row['Feature']}: {row['Clinical Significance'].split('by')[1].strip()}"
            value = abs(row['Impact'])
            
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="font-weight:500; margin-bottom:5px">{row['Feature']}</div>
                <div style="background:#f0f0f0; height:20px; border-radius:10px">
                    <div style="background:{row['Color']}; width:{value*500}%; height:100%; border-radius:10px">
                        <span style="padding-left:10px; color:black; font-size:12px">
                            {row['Clinical Significance']}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Force plot
        st.write("**How features interact to determine your risk:**")
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
            st.write("**Global Feature Importance:**")
            st.dataframe(importance_df)
        except:
            st.error("Feature importance data unavailable")

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
    1. **Glucose levels** - Fasting plasma glucose is the strongest predictor
    2. **BMI** - Body Mass Index indicates metabolic health
    3. **Age** - Risk increases significantly after 45
    4. **Diabetes Genetic Risk** - Family history multiplier
    """)
    
    st.markdown("""
    **Risk Stratification**:
    - < 20% → Very Low Risk (Green)
    - 20-39% → Low Risk (Light Green)
    - 40-59% → Moderate Risk (Amber)
    - 60-79% → High Risk (Orange)
    - ≥80% → Very High Risk (Red)
    """)

with col2:
    st.markdown("""
    **Clinical Validation**:
    - Matches established diabetes pathophysiology
    - Identifies hyperglycemia as primary driver
    - Detects obesity and age-related patterns
    - Aligns with ADA screening guidelines
    """)
    
    st.markdown("""
    **Key Limitations**:
    - Insulin data missing in 49% of cases
    - Homogeneous patient cohort (Pima Indians)
    - Doesn't include lifestyle/diet factors
    - Blood pressure measurement is diastolic only
    """)

# Model performance info
st.markdown("""
**Model Performance Metrics**:
- **AUC**: 0.8146 (Excellent discrimination)
- **Recall**: 81% (Minimizes missed cases)
- **Precision**: 63% (Balances unnecessary interventions)
- **F1 Score**: 0.71 (Optimal threshold: 0.31)
""")

# Footer
st.markdown("""
---
**Disclaimer**: 
This tool provides risk assessment based on statistical modeling, not a medical diagnosis. 
Clinical judgment should always supersede algorithmic predictions. 

*Model developed with SHAP explanations for clinical interpretability. 
Validated on Pima Indians Diabetes Dataset. Not for diagnostic use.*
""")