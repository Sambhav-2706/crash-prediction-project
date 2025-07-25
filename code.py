import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- LOAD MODEL AND ENCODERS ----
# These files must exist in the same directory as this script
model = joblib.load("aviation_damage_rf.pkl")
scaler = joblib.load("scaler.pkl")
le_target = joblib.load("label_encoder_damage.pkl")
feature_label_encoders = joblib.load("feature_label_encoders.pkl")

# ---- DEFINE FEATURE COLUMNS ----
# List in the exact order/columns used for model training
feature_columns = [
    # Replace with your actual feature names, for example:
    'Make', 'Model', 'Engine.Type', 'Weather.Condition',
    # ...add all the columns in correct order, as used in training!
    # Your CSV's column order MINUS dropped columns and target
]

st.title("Aviation Damage Severity Predictor")
st.write("Enter the incident details below. All fields are required.")

# ---- COLLECT USER INPUT ----
user_input = {}
for col in feature_columns:
    if col in feature_label_encoders:
        # For features trained with label encoding, use text input or a select box
        val = st.text_input(f"{col} (categorical)")
        user_input[col] = val
    else:
        # Assume numerical, offer number input
        val = st.number_input(f"{col} (numerical)", value=0)
        user_input[col] = val

if st.button("Predict Damage Severity"):

    # ---- FORM INPUT DATAFRAME ----
    input_df = pd.DataFrame([user_input])
    
    # ---- ENCODE CATEGORICAL FEATURES ----
    for col in feature_label_encoders:
        le = feature_label_encoders[col]
        # If user input unseen, assign -1 (or handle as desired)
        try:
            input_df[col] = le.transform([input_df[col][0]])
        except ValueError:
            input_df[col] = -1  # Unseen/new category

    # Ensure all columns are in correct numeric dtype
    input_df = input_df.astype(float)
    
    # ---- SCALE FEATURES ----
    input_scaled = scaler.transform(input_df)

    # ---- PREDICT ----
    pred = model.predict(input_scaled)
    pred_label = le_target.inverse_transform(pred)[0]

    st.success(f"Predicted Aircraft Damage Severity: **{pred_label}**")

    # Optionally, show probabilities
    pred_proba = model.predict_proba(input_scaled)
    classes = le_target.classes_
    st.write("Prediction probabilities:")
    proba_df = pd.DataFrame(pred_proba, columns=classes)
    st.dataframe(proba_df.style.format("{:.2%}"))
