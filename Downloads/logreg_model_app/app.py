# minimal_streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.title("Bank Account Prediction App")
st.write("Enter the details below to predict whether a user has a bank account.")

# Load the trained model safely
model_path = "logreg_model.joblib"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please make sure it is in the same folder as this app.")
    st.stop()

model = joblib.load(model_path)

# Get feature names dynamically
try:
    feature_names = model.feature_names_in_
except AttributeError:
    st.error("The loaded model does not contain feature names. Please retrain your model with scikit-learn >=0.24.")
    st.stop()

# Create input fields dynamically
user_input = {}
for feature in feature_names:
    if "_Yes" in feature or "_No" in feature or "_" in feature:  # binary or encoded categorical
        user_input[feature] = st.selectbox(f"{feature}", options=[0, 1])
    else:  # numeric features
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"The model predicts the user HAS a bank account (Probability: {prediction_proba:.2f})")
    else:
        st.warning(f"The model predicts the user DOES NOT have a bank account (Probability: {prediction_proba:.2f})")
