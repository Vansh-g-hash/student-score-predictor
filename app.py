import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.title("📊 Student Score Predictor")
st.write("Predict exam scores based on study hours.")

# User input
hours = st.number_input("Enter study hours:", min_value=0.0, max_value=12.0, step=0.5)

if st.button("Predict Score"):
    score = model.predict([[hours]])[0]
    st.success(f"📈 Predicted Score: {score:.2f}")

# Show example dataset
st.subheader("📌 Dataset Preview")
df = pd.read_csv("data/student_scores.csv")
st.write(df.head())
