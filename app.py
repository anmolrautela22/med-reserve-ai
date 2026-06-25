# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Med-Reserve AI", page_icon="🏥", layout="wide")

# Title and context
st.title("🏥 Med-Reserve: AI-Driven Patient Triage & Bed Resource Planner")
st.write("Real-time Emergency Room Resource Optimizer.")
st.markdown("---")

# Mock database tracking currently occupied hospital infrastructure
if 'icu_beds' not in st.session_state:
    st.session_state.icu_beds = 12
if 'general_beds' not in st.session_state:
    st.session_state.general_beds = 45


# Load our saved machine learning pipeline safely
@st.cache_resource
def load_ml_pipeline():
    with open('med_reserve_model.pkl', 'rb') as f:
        return pickle.load(f)


try:
    model = load_ml_pipeline()
except FileNotFoundError:
    st.error("🚨 Model file missing! Please execute 'python train_model.py' first in your terminal.")
    st.stop()

# Layout splits: UI Left Column for Triage Input | UI Right Column for Live State Analytics
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 New Patient Clinical Triage Input")

    # Form elements to gather features safely
    patient_id = st.text_input("Patient Unique ID / Aadhaar Hash", value="PAT-8839")
    age = st.slider("Patient Age", min_value=1, max_value=100, value=45)
    spo2 = st.slider("Oxygen Saturation (SpO2 %)", min_value=70, max_value=100, value=94)
    heart_rate = st.slider("Pulse Rate (BPM)", min_value=40, max_value=160, value=82)
    comorbidities = st.selectbox("Active Comorbidities Count (Diabetes, Hypertension, Chronic Kidney Disease, etc.)",
                                 [0, 1, 2, 3])

    submit_btn = st.button("Analyze & Reserve Bed Infrastructure", type="primary")

with col2:
    st.subheader("📊 Live Hospital Resource Status Dashboard")

    # Render operational status metrics blocks
    m_col1, m_col2 = st.columns(2)
    m_col1.metric(label="Available ICU Beds (Critical Care)", value=f"{st.session_state.icu_beds} / 15")
    m_col2.metric(label="Available Ward Beds (General/HDU)", value=f"{st.session_state.general_beds} / 60")

    st.markdown("---")

    if submit_btn:
        # Create input array mapping precisely to trained pipeline columns
        input_data = pd.DataFrame([[age, spo2, heart_rate, comorbidities]],
                                  columns=['Age', 'SpO2', 'Heart_Rate', 'Comorbidities'])

        # Call model to make an inference
        prediction = model.predict(input_data)[0]

        # Translate discrete classification levels into clinical designations
        if prediction == 2:
            triage_status = "CRITICAL: Urgent ICU Admission Required"
            estimated_stay = int(age * 0.1 + 10)  # Dynamic fallback heuristic calculation for demo
            st.error(f"🚨 Triage Risk Evaluation: {triage_status}")

            # Update resources dynamically during testing session
            if st.session_state.icu_beds > 0:
                st.session_state.icu_beds -= 1
                st.success("✅ Smart Allocation: Reserved 1 Intensive Care Unit (ICU) ventilator bed.")
            else:
                st.warning(
                    "⚠️ Critical Capacity Warning: No physical ICU beds free! Activating routing backup protocol.")

        elif prediction == 1:
            triage_status = "MODERATE: High Dependency Unit (HDU) Allocation"
            estimated_stay = int(age * 0.05 + 5)
            st.warning(f"⚠️ Triage Risk Evaluation: {triage_status}")

            if st.session_state.general_beds > 0:
                st.session_state.general_beds -= 1
                st.success("✅ Smart Allocation: Reserved 1 Monitoring bed in High Dependency Ward.")
        else:
            triage_status = "STABLE: General OPD / Routine Ward Stay"
            estimated_stay = np.random.randint(1, 4)
            st.info(f"ℹ️ Triage Risk Evaluation: {triage_status}")

            if st.session_state.general_beds > 0:
                st.session_state.general_beds -= 1
                st.success("✅ Smart Allocation: Assigned to General Ward space.")

        # Display estimated resource recovery metrics
        st.info(f"⏳ Estimated Length of Stay (LOS) based on profile: **{estimated_stay} Days**")