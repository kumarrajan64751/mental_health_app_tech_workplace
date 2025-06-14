
import streamlit as st
import numpy as np
import pickle
import json
import os
import pandas as pd  # ✅ Added for DataFrame conversion
from streamlit_lottie import st_lottie
from report_generator import generate_pdf

# ─── Session State ────────────────────────────────────────────────────────────
if "retake_triggered" not in st.session_state:
    st.session_state.retake_triggered = False
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# ─── Hide Streamlit Style ─────────────────────────────────────────────────────
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ─── Load Lottie Animations ───────────────────────────────────────────────────
def load_lottie(path: str):
    with open(path, "r") as f:
        return json.load(f)

greeting_anim = load_lottie("app/lottie_greeting.json")
healthy_anim = load_lottie("app/lottie_healthy.json")
support_anim = load_lottie("app/lottie_support.json")

if st.session_state.retake_triggered:
    st.session_state.retake_triggered = False
    st.session_state.prediction_made = False
    st.rerun()

# ─── Load Model and Label Encoders ────────────────────────────────────────────
with open("model/mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ─── Feature List (Order Matters) ─────────────────────────────────────────────
FEATURES = ['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
            'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
            'wellness_program', 'seek_help', 'anonymity', 'leave',
            'mental_health_consequence', 'phys_health_consequence',
            'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
            'mental_vs_physical', 'obs_consequence', 'Country']

# ─── Greeting Section ─────────────────────────────────────────────────────────
st_lottie(greeting_anim, height=250)
st.markdown("""
    <h1 style='text-align: center;'>
        <span style='color:#4B9CD3;'>🧠 NeuroLens</span>: Designed to support the minds that drive the world.
    </h1>
     <h4 style='text-align: center; color: gray;'>A smart mental health checkup built for today’s professionals.</h4>
     <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Name Input ───────────────────────────────────────────────────────────────
name = st.text_input("Enter your name:", "")
if not name:
    st.warning("Please enter your name to proceed.")
    st.stop()

st.markdown("### Please answer the following questions:")

# ─── User Input Form ──────────────────────────────────────────────────────────
user_responses = {}

user_responses['Age'] = st.slider("What is your age?", 18, 100, 25)
user_responses['Gender'] = st.selectbox("What is your gender?", ["Male", "Female", "Other"])
user_responses['self_employed'] = st.selectbox("Are you self-employed?", ["Yes", "No"])
user_responses['family_history'] = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
user_responses['work_interfere'] = st.selectbox("Does your mental health interfere with your work?", ["Never", "Rarely", "Sometimes", "Often"])
user_responses['no_employees'] = st.selectbox("How many employees are in your company?", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
user_responses['remote_work'] = st.selectbox("Do you work remotely?", ["Yes", "No"])
user_responses['tech_company'] = st.selectbox("Do you work in a tech company?", ["Yes", "No"])
user_responses['benefits'] = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
user_responses['care_options'] = st.selectbox("Do you know the mental health care options provided by your employer?", ["Yes", "No", "Not sure"])
user_responses['wellness_program'] = st.selectbox("Has your employer ever discussed mental health as part of a wellness program?", ["Yes", "No", "Don't know"])
user_responses['seek_help'] = st.selectbox("Does your employer provide resources to seek help for mental health?", ["Yes", "No", "Don't know"])
user_responses['anonymity'] = st.selectbox("Is anonymity provided for mental health services?", ["Yes", "No", "Don't know"])
user_responses['leave'] = st.selectbox("How easy is it to take mental health leave?", ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
user_responses['mental_health_consequence'] = st.selectbox("Do you think there would be consequences of discussing mental health at work?", ["Yes", "No", "Maybe"])
user_responses['phys_health_consequence'] = st.selectbox("Do you think there would be consequences of discussing physical health at work?", ["Yes", "No", "Maybe"])
user_responses['coworkers'] = st.selectbox("Are you comfortable discussing mental health with coworkers?", ["Yes", "No", "Some of them"])
user_responses['supervisor'] = st.selectbox("Are you comfortable discussing mental health with your supervisor?", ["Yes", "No", "Some of them"])
user_responses['mental_health_interview'] = st.selectbox("Would you bring up a mental health issue in a job interview?", ["Yes", "No", "Maybe"])
user_responses['phys_health_interview'] = st.selectbox("Would you bring up a physical health issue in a job interview?", ["Yes", "No", "Maybe"])
user_responses['mental_vs_physical'] = st.selectbox("Do you think mental health is treated the same as physical health?", ["Yes", "No", "Don't know"])
user_responses['obs_consequence'] = st.selectbox("Have you observed consequences of mental health issues in the workplace?", ["Yes", "No"])
user_responses['Country'] = st.selectbox("Which country are you currently residing in?", ["United States", "India", "United Kingdom", "Canada", "Germany", "Other"])

# ─── Prediction ───────────────────────────────────────────────────────────────
if st.button("Submit Assessment"):
    input_dict = {}
    for col in FEATURES:
        val = user_responses[col]
        if col in label_encoders:
            input_dict[col] = label_encoders[col].transform([val])[0]
        else:
            input_dict[col] = val

    input_df = pd.DataFrame([input_dict])  # ✅ Use DataFrame instead of np.array

    prediction = model.predict(input_df)[0]

    # Reverse label (optional, use for displaying original label)
    target_encoder = label_encoders.get('__target__', None)
    if target_encoder:
        prediction_label = target_encoder.inverse_transform([prediction])[0]
    else:
        prediction_label = "Yes" if prediction == 1 else "No"

    st.session_state.prediction_made = True
    st.markdown("---")

    if prediction == 1:
        st.markdown("### 🧠 You may **Need Mental Health Support**")
        st_lottie(support_anim, height=300)
        st.info("It looks like you might be experiencing some challenges related to mental health, such as anxiety, stress, or depression. "
                "You're not alone—it's okay to ask for help. Please consider reaching out to a mental health professional.")
        st.markdown("[🧘 Mindfulness Exercises](#)")
        st.markdown("[🩺 Consult a Doctor](#)")
    else:
        st.markdown("### ✅ You are **Mentally Healthy**")
        st_lottie(healthy_anim, height=300)
        st.success("Well done! You have shown signs of maintaining good mental health. Keep taking care of yourself and others!")

    # PDF Report
    st.markdown("### 📄 Download Your Report")
    if st.download_button(
        label="Download PDF Report",
        data=generate_pdf(name, user_responses['Age'], user_responses, prediction_label),
        file_name=f"{name}_mental_health_report.pdf",
        mime="application/pdf"
    ):
        st.success("Your PDF report has been downloaded.")

# ─── Retake Option ────────────────────────────────────────────────────────────
if st.session_state.prediction_made:
    if st.button("🔁 Retake"):
        st.session_state.retake_triggered = True
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
st.markdown("""
    <hr style="margin-top: 30px; margin-bottom: 10px;">
    <p style='text-align: center; color: gray; font-size: 20px;'>
        Made with ❤️ to support your mental well-being
    </p>
""", unsafe_allow_html=True)
