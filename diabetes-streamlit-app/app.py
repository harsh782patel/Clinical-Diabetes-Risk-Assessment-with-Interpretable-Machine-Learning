import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Clinical Diabetes Risk Assessment", page_icon="🩺", layout="wide")

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
st.title('🩺 Clinical Diabetes Risk Assessment')
st.markdown("""
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

def create_risk_scale(probability):
    # Simple, clear risk levels
    risk_levels = [
        {"min": 0, "max": 0.2, "label": "Very Low", "color": "#4CAF50"},
        {"min": 0.2, "max": 0.4, "label": "Low", "color": "#8BC34A"},
        {"min": 0.4, "max": 0.6, "label": "Moderate", "color": "#FFC107"},
        {"min": 0.6, "max": 0.8, "label": "High", "color": "#FF9800"},
        {"min": 0.8, "max": 1.0, "label": "Very High", "color": "#F44336"}
    ]
    
    current_risk = next((level for level in risk_levels 
                       if level["min"] <= probability < level["max"]), risk_levels[-1])
    
    # Create a super simple bar chart
    fig = go.Figure()
    
    # Add main risk bar
    fig.add_trace(go.Bar(
        x=["Your Risk"],
        y=[probability*100],
        marker_color=current_risk["color"],
        width=0.6,
        text=[f"{probability:.1%}"],
        textposition='outside',
        textfont=dict(size=24, color=current_risk["color"]),
        hoverinfo='none'
    ))
    
    # Add reference lines
    for level in risk_levels:
        fig.add_shape(
            type="line",
            x0=-0.5, x1=0.5,
            y0=level["min"]*100, y1=level["min"]*100,
            line=dict(color="#AAAAAA", width=1, dash="dot")
        )
    
    # Add risk level labels on right side
    fig.add_trace(go.Scatter(
        x=[0.5]*5,
        y=[10, 30, 50, 70, 90],
        text=[level["label"] for level in risk_levels],
        mode="text",
        textfont=dict(color=[level["color"] for level in risk_levels], size=14),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Clean, minimal layout
    fig.update_layout(
        height=400,
        yaxis=dict(
            title="Risk Percentage",
            range=[0, 100],
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            tickfont=dict(size=12),
            titlefont=dict(size=14)
        ),
        xaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=120),
        hovermode=False
    )
    
    return fig, current_risk

# Predict and display results
if st.button('Assess Diabetes Risk', type="primary", use_container_width=True):
    # Predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    
    # Enhanced risk stratification
    if probability < 0.2:
        risk_category = "Very Low Risk"
        color = "#2ecc71"  # Green
        emoji = "✅"
        risk_level = 1
    elif probability < 0.4:
        risk_category = "Low Risk"
        color = "#27ae60"  # Light green
        emoji = "🟢"
        risk_level = 2
    elif probability < 0.6:
        risk_category = "Moderate Risk"
        color = "#f39c12"  # Amber
        emoji = "🟠"
        risk_level = 3
    elif probability < 0.8:
        risk_category = "High Risk"
        color = "#e74c3c"  # Red
        emoji = "🔴"
        risk_level = 4
    else:
        risk_category = "Very High Risk"
        color = "#c0392b"  # Dark red
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
    
    # Create enhanced risk scale
    st.write("**Risk Assessment Scale:**")
    risk_fig, current_risk = create_risk_scale(probability)
    st.plotly_chart(risk_fig, use_container_width=True)
    
    # Risk interpretation using native Streamlit components
    risk_text = {
        "Very Low": "Your risk is very low. Keep up healthy habits!",
        "Low": "You have slightly elevated risk. Consider lifestyle improvements.",
        "Moderate": "Moderate risk detected. We recommend preventive actions.",
        "High": "High risk identified. Medical consultation advised.",
        "Very High": "Very high risk. Please consult a doctor soon."
    }
    
    with st.container():
        st.subheader(f"{current_risk['label']} Risk ({probability:.1%})")
        st.write(risk_text[current_risk['label']])
        
        # Create risk scale visualization using Matplotlib
        st.write("**RISK SCALE**")
        
        # Create gradient color scale
        colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
        cmap = mcolors.LinearSegmentedColormap.from_list("risk_scale", colors)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 1))
        fig.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Create gradient bar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 100, 0, 1])
        
        # Add current risk marker
        ax.plot(probability * 100, 0.5, 'ko', markersize=8)
        
        # Set axis properties
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        
        # Add labels
        ax.text(0, 1.5, "Low", ha='left', va='center', fontsize=10)
        ax.text(100, 1.5, "High", ha='right', va='center', fontsize=10)
        ax.text(probability * 100, -0.5, f"Your risk: {probability:.1%}", 
                ha='center', va='top', fontsize=12, fontweight='bold')
        
        st.pyplot(fig, use_container_width=True)
    
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
        
        # Create horizontal bar chart with Plotly
        st.subheader("Risk Factor Importance")
        
        impact_df = impact_df.sort_values('Importance', ascending=True)
        fig = go.Figure(go.Bar(
            x=impact_df['Importance'],
            y=impact_df['Feature'],
            orientation='h',
            marker=dict(color='#4CAF50')
        ))
        
        fig.update_layout(
            height=500,
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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