import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import os

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

# App title with improved styling
st.set_page_config(page_title="Clinical Diabetes Risk Assessment", page_icon="🩺", layout="wide")
st.title('🩺 Clinical Diabetes Risk Assessment')
st.write("""
<div style='font-size:18px; padding-bottom:15px'>
<i>Interpretable ML model predicting diabetes risk using routine clinical measurements</i>
</div>
""", unsafe_allow_html=True)

# Sidebar inputs with enhanced UI
with st.sidebar:
    st.header('Patient Clinical Measurements')
    
    # BMI calculator section
    st.subheader("BMI Calculator", help="Calculate BMI from height and weight")
    bmi_method = st.radio("BMI Input Method:", ["Enter BMI directly", "Calculate from height/weight"])
    
    bmi_value = 26.2  # Default value
    if bmi_method == "Calculate from height/weight":
        col1, col2 = st.columns(2)
        with col1:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=75)
        
        if height_cm > 0 and weight_kg > 0:
            bmi_value = weight_kg / ((height_cm/100) ** 2)
            st.success(f"Calculated BMI: **{bmi_value:.1f}** kg/m²")
            bmi_category = "Healthy" if bmi_value < 25 else "Overweight" if bmi_value < 30 else "Obese"
            st.write(f"**Category:** {bmi_category}")
    else:
        bmi_value = st.slider('BMI (kg/m²)', 15.0, 50.0, 26.2, step=0.1, 
                              help=CLINICAL_DEFINITIONS['BMI'])
    
    st.divider()
    
    # Add clinical definitions expander
    with st.expander("ℹ️ Measurement Definitions"):
        for feature, definition in CLINICAL_DEFINITIONS.items():
            st.markdown(f"**{feature}**: {definition}")
    
    # Explain blood pressure measurement
    st.markdown("""
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
    inputs['BMI'] = bmi_value  # Use calculated or direct value
    inputs['DiabetesPedigreeFunction'] = st.sidebar.slider('Diabetes Genetic Risk', 0.08, 2.5, 0.5, step=0.01, 
                                                          help=CLINICAL_DEFINITIONS['DiabetesPedigreeFunction'])
    inputs['Age'] = st.sidebar.slider('Age (years)', 20, 85, 35, 
                                     help=CLINICAL_DEFINITIONS['Age'])
    
    return pd.DataFrame([inputs], columns=feature_names)

# Get user input
input_df = user_input_features()

# Display user inputs in cards
st.subheader('Patient Input Features')

# Create card layout for inputs
cols = st.columns(4)
metrics = [
    ("Glucose", f"{input_df['Glucose'].iloc[0]} mg/dL", "#FF6B6B"),
    ("Blood Pressure", f"{input_df['BloodPressure'].iloc[0]} mmHg", "#4ECDC4"),
    ("BMI", f"{input_df['BMI'].iloc[0]:.1f} kg/m²", "#FFD166"),
    ("Age", f"{input_df['Age'].iloc[0]} years", "#6A0572")
]

for i, (title, value, color) in enumerate(metrics):
    with cols[i]:
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; 
                    text-align:center; background-color:{color}10; 
                    border-left: 4px solid {color}; margin-bottom:15px">
            <div style="font-weight:bold; font-size:16px; color:{color}">{title}</div>
            <div style="font-size:24px; margin:10px 0">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# Additional metrics in another row
cols = st.columns(4)
metrics = [
    ("Pregnancies", input_df['Pregnancies'].iloc[0], "#1A535C"),
    ("Skin Thickness", f"{input_df['SkinThickness'].iloc[0]} mm", "#FF9F1C"),
    ("Insulin", f"{input_df['Insulin'].iloc[0]} μU/mL", "#5E548E"),
    ("Genetic Risk", f"{input_df['DiabetesPedigreeFunction'].iloc[0]:.2f}", "#E71D36")
]

for i, (title, value, color) in enumerate(metrics):
    with cols[i]:
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; 
                    text-align:center; background-color:{color}10; 
                    border-left: 4px solid {color}; margin-bottom:15px">
            <div style="font-weight:bold; font-size:16px; color:{color}">{title}</div>
            <div style="font-size:24px; margin:10px 0">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced risk visualization function - FIXED VERSION
def create_risk_meter(probability):
    # Create compact figure
    fig, ax = plt.subplots(figsize=(9, 3), facecolor='#f8f9fa')
    fig.subplots_adjust(top=0.7, bottom=0.3)
    
    # Risk categories with enhanced descriptions
    risk_levels = [
        {"name": "Very Low", "min": 0.0, "max": 0.2, "color": "#4CAF50", "icon": "✅"},
        {"name": "Low", "min": 0.2, "max": 0.4, "color": "#8BC34A", "icon": "🟢"},
        {"name": "Moderate", "min": 0.4, "max": 0.6, "color": "#FFC107", "icon": "🟠"},
        {"name": "High", "min": 0.6, "max": 0.8, "color": "#FF9800", "icon": "🔴"},
        {"name": "Very High", "min": 0.8, "max": 1.0, "color": "#F44336", "icon": "⚠️"}
    ]
    
    # Find current risk level
    current_risk = next((level for level in risk_levels if level["min"] <= probability < level["max"]), risk_levels[-1])
    
    # Create gauge background
    for level in risk_levels:
        ax.barh(0, level["max"] - level["min"], left=level["min"], 
                height=0.3, color=level["color"], alpha=0.8)
        
        # Add risk level name at the bottom
        center = (level["min"] + level["max"]) / 2
        ax.text(center, -0.2, level["name"], 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add threshold markers
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        ax.plot([threshold, threshold], [-0.05, 0], 'k-', linewidth=1, alpha=0.5)
    
    # Add current risk indicator
    ax.plot([probability, probability], [0, 0.3], 'k-', linewidth=2)
    ax.plot(probability, 0.3, 'k^', markersize=12)
    
    # Add current risk value
    ax.text(probability, 0.35, f'Current Risk: {probability:.1%}', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))
    
    # Add clinical interpretation
    ax.text(0.5, 0.5, f"CLINICAL INTERPRETATION: {current_risk['name'].upper()} RISK {current_risk['icon']}", 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=current_risk['color'], alpha=0.2))
    
    # Add risk scale label
    ax.text(0, 0.45, "LOW RISK", ha='left', va='center', fontsize=8, color='#4CAF50')
    ax.text(1, 0.45, "HIGH RISK", ha='right', va='center', fontsize=8, color='#F44336')
    
    # Remove axes decoration
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.6)
    ax.axis('off')
    
    return fig

# Predict and display results
if st.button('Assess Diabetes Risk', type="primary", use_container_width=True):
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
    
    # Create enhanced risk meter
    st.write("**Risk Assessment Scale:**")
    risk_fig = create_risk_meter(probability)
    st.pyplot(risk_fig)
    
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
        
        # Create horizontal bar chart
        st.subheader("Risk Factor Importance")
        fig, ax = plt.subplots(figsize=(10, 5))
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
    <div style="border-radius:10px; padding:20px; background-color:#e3f2fd; margin-bottom:20px">
        <h3>Dominant Risk Factors</h3>
        <ol style="font-size:16px">
            <li>Glucose levels</li>
            <li>BMI</li>
            <li>Age</li>
            <li>Diabetes Genetic Risk</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="border-radius:10px; padding:20px; background-color:#fff8e1; margin-bottom:20px">
        <h3>Risk Stratification</h3>
        <ul style="font-size:16px">
            <li>< 20% → Very Low Risk</li>
            <li>20-39% → Low Risk</li>
            <li>40-59% → Moderate Risk</li>
            <li>60-79% → High Risk</li>
            <li>≥80% → Very High Risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="border-radius:10px; padding:20px; background-color:#e8f5e9; margin-bottom:20px">
        <h3>Clinical Validation</h3>
        <ul style="font-size:16px">
            <li>Matches established diabetes pathophysiology</li>
            <li>Identifies hyperglycemia as primary driver</li>
            <li>Detects obesity and age-related patterns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="border-radius:10px; padding:20px; background-color:#ffebee; margin-bottom:20px">
        <h3>Key Limitations</h3>
        <ul style="font-size:16px">
            <li>Insulin data missing in 49% of cases</li>
            <li>Homogeneous patient cohort</li>
            <li>Doesn't include lifestyle factors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Model performance info
st.markdown("""
<div style="border-radius:10px; padding:20px; background-color:#f5f5f5; margin-bottom:20px">
    <h3>Model Performance Metrics</h3>
    <ul style="font-size:16px">
        <li>AUC: 0.8146</li>
        <li>Recall: 81%</li>
        <li>Precision: 63%</li>
        <li>F1 Score: 0.71</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style="font-size:14px; color:#666; text-align:center">
    <strong>Disclaimer</strong>: 
    This tool provides risk assessment based on statistical modeling, not a medical diagnosis. 
    Clinical judgment should always supersede algorithmic predictions. 
    <br><br>
    <i>Model developed for clinical interpretability. Validated on Pima Indians Diabetes Dataset.</i>
</div>
""", unsafe_allow_html=True)