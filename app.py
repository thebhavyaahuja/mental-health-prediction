import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load trained models and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
nn_model = tf.keras.models.load_model('nn_model.h5')


st.set_page_config("Mental Health Risk Predictor",layout = "centered")
st.title("Mental Health Risk Predictor")
st.markdown("Fill in the details below to predict your mental health risk level.")

# Define input fields
# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student"])
work_environment = st.selectbox("Work Environment", ["Supportive", "Neutral", "Unsupportive"])
mental_health_history = st.selectbox("Previous Mental Health Diagnosis?", ["Yes", "No"])
seeks_treatment = st.selectbox("Currently Seeking Treatment?", ["Yes", "No"])

if st.button("Predict Risk Level"):
    # Prepare input dataframe
    input_dict = {
        "age": [age],
        "gender_Male": [1 if gender == "Male" else 0],
        "gender_Other": [1 if gender == "Other" else 0],
        "employment_status_Student": [1 if employment_status == "Student" else 0],
        "employment_status_Unemployed": [1 if employment_status == "Unemployed" else 0],
        "work_environment_Neutral": [1 if work_environment == "Neutral" else 0],
        "work_environment_Unsupportive": [1 if work_environment == "Unsupportive" else 0],
        "mental_health_history_Yes": [1 if mental_health_history == "Yes" else 0],
        "seeks_treatment_Yes": [1 if seeks_treatment == "Yes" else 0],
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_dict)

    # Make sure all model features are covered (fill missing with 0)
    all_features = scaler.feature_names_in_
    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[all_features]

    # Scale features
    X_scaled = scaler.transform(input_df)

    # Predict with Random Forest
    rf_pred = rf_model.predict(X_scaled)[0]
    rf_proba = rf_model.predict_proba(X_scaled)[0]

    # Predict with Neural Network
    nn_proba = nn_model.predict(X_scaled)[0]
    nn_pred = np.argmax(nn_proba)

    # Map numeric predictions to labels
    label_map = {0: "Low", 1: "Moderate", 2: "High"}

    # Display results
    st.subheader("Model Predictions")

    st.write(f"Random Forest Prediction: {label_map[rf_pred]}")
    st.write(f"Class Probabilities (RF): {dict(zip(['Low', 'Moderate', 'High'], rf_proba.round(2)))}")

    st.write(f"Neural Network Prediction: {label_map[nn_pred]}")
    st.write(f"Class Probabilities (NN): {dict(zip(['Low', 'Moderate', 'High'], nn_proba.round(2)))}")

    # Final risk: average prediction
    avg_pred = np.mean([rf_pred, nn_pred])
    final_pred = round(avg_pred)
    st.success(f"Overall Risk Level:{label_map[final_pred]}")
