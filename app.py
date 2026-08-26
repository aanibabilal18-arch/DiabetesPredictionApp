import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's medical information below.")

# Model accuracy
st.info("KNN Model Accuracy: 71%")

st.divider()

# Input fields
pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=150.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=79.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.000,
    max_value=3.000,
    value=0.471,
    format="%.3f"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

st.divider()

# Prediction
if st.button("Predict Diabetes"):

    # Create input array
    sample = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Scale input
    sample_scaled = scaler.transform(sample)

    # Make prediction
    prediction = model.predict(sample_scaled)[0]

    # Get probabilities
    probability = model.predict_proba(sample_scaled)[0]

    # Display probabilities
    st.write(
        f"Estimated probability of No Diabetes: "
        f"**{probability[0] * 100:.1f}%**"
    )

    st.write(
        f"Estimated probability of Diabetes: "
        f"**{probability[1] * 100:.1f}%**"
    )

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")

    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical advice.")
