import streamlit as st
import pandas as pd
import joblib

# Load trained model
model_data = joblib.load("model.pkl")
model = model_data["model"]
columns = model_data["columns"]

# Page settings
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰"
)

st.title("💰 Insurance Cost Prediction")
st.write("Enter your details to estimate your insurance cost.")

# User inputs
age = st.number_input("Age", min_value=1, max_value=100, value=25)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Prediction button
if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Convert categorical data
    input_data = pd.get_dummies(
        input_data,
        columns=["sex", "smoker", "region"],
        drop_first=True
    )

    # Make sure columns match training data
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Insurance Cost: ₹{prediction:,.2f}"
    )