import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("salary_model.pkl")
edu_encoder = joblib.load("edu_encoder.pkl")
job_encoder = joblib.load("job_encoder.pkl")
loc_encoder = joblib.load("loc_encoder.pkl")

st.title("💼 Employee Salary Predictor")

# Input fields
experience = st.slider("Years of Experience", 0, 40, 5)
education = st.selectbox("Education Level", list(edu_encoder.classes_))
job = st.selectbox("Job Title", list(job_encoder.classes_))
location = st.selectbox("Location", list(loc_encoder.classes_))

# Encode inputs safely
edu_num = edu_encoder.transform([education])[0]
job_num = job_encoder.transform([job])[0]
loc_num = loc_encoder.transform([location])[0]

input_data = pd.DataFrame([[experience, edu_num, job_num, loc_num]], 
                          columns=['Experience', 'Education_Level', 'Job_Title', 'Location'])

# Predict and display
if st.button("Predict Salary"):
    result = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹{int(result):,}")