import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('placement_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Campus Placement Prediction App")
st.markdown("Predict whether a student will be placed based on their details.")

# Input features
gender = st.selectbox("Gender", ["Male", "Female"])
stream = st.selectbox("Stream", ["Electronics And Communication", "Computer Science",
                                 "Information Technology", "Mechanical", "Electrical", "Civil"])
internships = st.slider("Number of Internships", 0, 3, step=1)
cgpa = st.slider("CGPA", 5.0, 10.0, step=0.1)
age = st.number_input("Age", min_value=18, max_value=30, value=21, step=1)
hostel = st.selectbox("Residing in Hostel?", ["Yes", "No"])
history_of_backlogs = st.selectbox("History of Backlogs?", ["Yes", "No"])

# Encode inputs to match model
gender_encoded = 1 if gender == "Male" else 0
hostel_encoded = 1 if hostel == "Yes" else 0
backlogs_encoded = 1 if history_of_backlogs == "Yes" else 0

# One-hot encoding for stream
stream_options = ["Electronics And Communication", "Computer Science",
                  "Information Technology", "Mechanical", "Electrical", "Civil"]
stream_encoded = [1 if stream == option else 0 for option in stream_options[1:]]  # Drop first category

# Combine all features
features = [age, gender_encoded, internships, cgpa, hostel_encoded, backlogs_encoded] + stream_encoded
features = np.array(features).reshape(1, -1)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("The student is likely to be placed!")
    else:
        st.error("The student is unlikely to be placed.")
