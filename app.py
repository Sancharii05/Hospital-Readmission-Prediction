import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_extras.metric_cards import style_metric_cards



# Load model
model = pickle.load(open("xgboost_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Hospital Readmission Prediction", layout="wide")

# Title and introduction
st.title("🏥 Hospital Readmission Prediction for Patients (30 Day Window)")
st.markdown("""
Hospital readmission within 30 days is a critical metric in healthcare, often used to assess the quality of care and the effectiveness of discharge planning.
""")

# Sidebar for inputs
st.sidebar.header("Patient Input Details")
with st.sidebar.form("patient_form"):
    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=0, max_value=30, value=5)
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=150, value=40)
    num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=1)
    num_medications = st.number_input("Number of Medications", min_value=0, max_value=100, value=20)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=20, value=5)

    age_range = st.selectbox("Age Range",
                             ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"])
    age_midpoint = (int(age_range.split('-')[0]) + int(age_range.split('-')[1])) / 2

    disposition_options = {
        "1: Discharged to home": 1,
        "2: To another short term hospital": 2,
        "3: To SNF (Skilled Nursing Facility)": 3,
        "4: To ICF (Intermediate Care Facility)": 4,
        "5: To another inpatient care": 5,
        "6: Home with health service": 6,
        "7: Left AMA": 7,
        "8: Home under IV care": 8,
        "10: To short term general hospital": 10,
        "13: To psychiatric hospital": 13,
        "14: To rehab facility": 14,
        "15: Long-term care hospital": 15,
        "18: Hospice (home)": 18,
        "19: Hospice (medical)": 19,
        "21: Still patient / outpatient return": 21,
        "23: Long term acute care hospital": 23,
        "24: Nursing facility": 24,
        "27: Federal facility": 27,
        "28: Cancer/Children hospital": 28
    }
    discharge_label = st.selectbox("Discharge Disposition", list(disposition_options.keys()))
    discharge_id = disposition_options[discharge_label]

    diabetes_med = st.radio("Is the patient on diabetes medication?", ['Yes', 'No'])

    diag_mapping = {
        "1: Circulatory problems (like heart issues)": 1.0,
        "2: Respiratory (lungs)": 2.0,
        "3: Digestive (stomach)": 3.0,
        "4: Diabetes": 4.0,
        "5: Injuries": 5.0,
        "6: Bones/joints": 6.0,
        "7: Kidney/bladder": 7.0,
        "8: Cancer/tumors": 8.0
    }

    diag1 = diag_mapping[st.selectbox("Diagnosis 1 (Primary Group)", list(diag_mapping.keys()))]
    diag2 = diag_mapping[st.selectbox("Diagnosis 2 (Secondary Group)", list(diag_mapping.keys()))]
    diag3 = diag_mapping[st.selectbox("Diagnosis 3 (Tertiary Group)", list(diag_mapping.keys()))]

    submitted = st.form_submit_button("Predict")

# Prediction section (top after header)
if submitted:
    input_dict = {
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_diagnoses': number_diagnoses,
        f'age_{age_midpoint}': 1,
        f'discharge_disposition_id_{discharge_id}': 1,
        'diabetesMed_Yes': 1 if diabetes_med == 'Yes' else 0,
        f'level1_diag1_{diag1}': 1,
        f'level1_diag2_{diag2}': 1,
        f'level1_diag3_{diag3}': 1
    }

    all_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
                    'age_15.0', 'age_25.0', 'age_35.0', 'age_45.0', 'age_55.0', 'age_65.0', 'age_75.0', 'age_85.0',
                    'age_95.0',
                    'discharge_disposition_id_2', 'discharge_disposition_id_7', 'discharge_disposition_id_10',
                    'discharge_disposition_id_11',
                    'discharge_disposition_id_18', 'discharge_disposition_id_19', 'discharge_disposition_id_20',
                    'discharge_disposition_id_27', 'discharge_disposition_id_28', 'diabetesMed_Yes',
                    'level1_diag1_1.0', 'level1_diag1_2.0', 'level1_diag1_3.0', 'level1_diag1_4.0', 'level1_diag1_5.0',
                    'level1_diag1_6.0', 'level1_diag1_7.0', 'level1_diag1_8.0', 'level1_diag2_1.0', 'level1_diag2_2.0',
                    'level1_diag2_3.0', 'level1_diag2_4.0', 'level1_diag2_5.0', 'level1_diag2_6.0', 'level1_diag2_7.0',
                    'level1_diag2_8.0', 'level1_diag3_1.0', 'level1_diag3_2.0', 'level1_diag3_3.0', 'level1_diag3_4.0',
                    'level1_diag3_5.0', 'level1_diag3_6.0', 'level1_diag3_7.0', 'level1_diag3_8.0']

    final_input = {feat: input_dict.get(feat, 0) for feat in all_features}
    input_df = pd.DataFrame([final_input])

    prediction = model.predict(input_df)[0]

    # Result display
    st.markdown("""
    <div style='padding: 1.5rem; background-color: #f0f2f6; border-radius: 0.6rem; border-left: 8px solid #1f77b4;'>
        <h2 style='color:#1f77b4;'>🩺 Prediction Result</h2>
        <p style='font-size: 1.3rem;'>
        <strong>Result:</strong> <span style='color: {};'> <b>{}</b> </span>
        </p>
    </div>
    """.format("#d62728" if prediction == 1 else "#2ca02c",
               "⚠️ The patient is likely to be readmitted within 30 days." if prediction == 1 else "✅ The patient is not likely to be readmitted within 30 days."),
                unsafe_allow_html=True)


# Static: About the model with metrics
with st.expander("📘 About This Model"):
    st.markdown("""
    We use the **XGBoost Classifier**, a powerful gradient boosting algorithm well-suited for classification tasks on structured data.

    ### Model Performance Metrics:
    """)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "59%", help="Model is balanced")
    col2.metric("Precision", "15%")
    col3.metric("Recall", "56%", help="High Recall preferred")
    col4.metric("F1 Score", "24%")
    style_metric_cards(border_left_color="#4CAF50")

    st.markdown("""
    - **Why is Recall emphasized?**
        - Because **false negatives** (missed readmission cases) are more critical in healthcare.
        - High recall ensures we **flag as many potential readmissions as possible**.

    This model is trained on real hospital data to identify **patients likely to be readmitted within 30 days**.
    """)

# Static: Why this problem matters
with st.expander("🌍 Why This Problem Matters"):
    st.markdown("""
    - **For Hospitals**: Helps identify high-risk patients and allocate resources more effectively (e.g., follow-ups, social support).
    - **For Patients**: Enables better care planning, reducing risk of complications.
    - **For Policy Makers**: Supports initiatives aimed at improving care quality and reducing unnecessary healthcare expenditures.
    """)
