import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="🧠",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/rf_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')  # Only target encoder
        
        return rf_model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Simple prediction function
def predict_risk(data):
    rf_model, scaler, label_encoder = load_models()
    
    if rf_model is None:
        return "Error loading models"
    
    # Prepare input data exactly like in notebook
    input_df = pd.DataFrame([{
        'age': data['age'],
        'gender': data['gender'].lower(),
        'employment_status': data['employment_status'].lower(),
        'work_environment': data['work_environment'].lower(), 
        'mental_health_history': data['mental_health_history'].lower(),
        'seeks_treatment': data['seeks_treatment'].lower(),
        'stress_level': data['stress_level'],
        'sleep_hours': data['sleep_hours'],
        'physical_activity_days': data['physical_activity_days'],
        'depression_score': data['depression_score'],
        'anxiety_score': data['anxiety_score'],
        'social_support_score': data['social_support_score'],
        'productivity_score': data['productivity_score']
    }])
    
    # One-hot encode categorical variables (same as notebook)
    categorical_cols = ['gender', 'employment_status', 'work_environment', 'mental_health_history', 'seeks_treatment']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Get the feature names from training (you'll need to save these)
    # For now, create expected columns based on your training data
    expected_columns = rf_model.feature_names_in_  # This should work if sklearn >= 0.24
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training order
    input_encoded = input_encoded[expected_columns]
    
    # Scale numerical features
    numerical_cols = ['age', 'stress_level', 'sleep_hours', 'physical_activity_days',
                      'depression_score', 'anxiety_score', 'social_support_score', 'productivity_score']
    
    # Only scale columns that exist and are numerical
    cols_to_scale = [col for col in numerical_cols if col in input_encoded.columns]
    input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])
    
    # Predict
    prediction = rf_model.predict(input_encoded)[0]
    probabilities = rf_model.predict_proba(input_encoded)[0]
    
    # Get risk label using the target label encoder
    risk_labels = ['High', 'Low', 'Medium']  # Based on your encoding order
    risk_label = risk_labels[prediction]
    
    return risk_label, probabilities

# Main app
st.title("🧠 Mental Health Risk Assessment")

# Input form
with st.form("assessment"):
    st.subheader("Personal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 80, 25)
        gender = st.selectbox("Gender", ["Male", "Female","Non"])
        employment_status = st.selectbox("Employment", ["Employed", "Unemployed", "Self-Employed"])
    
    with col2:
        work_environment = st.selectbox("Work Environment", ["Remote", "On-site", "Hybrid"])
        mental_health_history = st.selectbox("Mental Health History", ["Yes", "No"])
        seeks_treatment = st.selectbox("Seeks Treatment", ["Yes", "No"])
    
    st.subheader("Health Metrics")
    
    col3, col4 = st.columns(2)
    with col3:
        stress_level = st.slider("Stress Level", 1, 10, 5)
        sleep_hours = st.slider("Sleep Hours", 3, 12, 7)
        physical_activity_days = st.slider("Exercise Days/Week", 0, 7, 3)
    
    with col4:
        depression_score = st.slider("Depression Score", 0, 30, 10)
        anxiety_score = st.slider("Anxiety Score", 0, 21, 7)
        social_support_score = st.slider("Social Support", 0, 100, 50)
        productivity_score = st.slider("Productivity", 0, 100, 70)
    
    submitted = st.form_submit_button("Assess Risk")

# Show results
if submitted:
    data = {
        'age': age,
        'gender': gender,
        'employment_status': employment_status,
        'work_environment': work_environment,
        'mental_health_history': mental_health_history,
        'seeks_treatment': seeks_treatment,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'physical_activity_days': physical_activity_days,
        'depression_score': depression_score,
        'anxiety_score': anxiety_score,
        'social_support_score': social_support_score,
        'productivity_score': productivity_score
    }
    
    try:
        risk_label, probabilities = predict_risk(data)
        
        st.success("Assessment Complete!")
        
        # Display result
        if risk_label == "Low":
            st.success(f"🟢 Mental Health Risk: **{risk_label}**")
        elif risk_label == "Medium":
            st.warning(f"🟡 Mental Health Risk: **{risk_label}**")
        else:
            st.error(f"🔴 Mental Health Risk: **{risk_label}**")
        
        # Show probabilities
        st.subheader("Model Confidence")
        col1, col2, col3 = st.columns(3)
        
        # Map probabilities to correct labels
        prob_dict = dict(zip(['High', 'Low', 'Medium'], probabilities))
        
        with col1:
            st.metric("Low Risk", f"{prob_dict['Low']*100:.1f}%")
        with col2:
            st.metric("Medium Risk", f"{prob_dict['Medium']*100:.1f}%") 
        with col3:
            st.metric("High Risk", f"{prob_dict['High']*100:.1f}%")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")