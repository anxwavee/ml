import streamlit as st
import requests

st.title("Iris Classifier")
st.write("Enter flower measurements to get a prediction")

sepal_length = st.number_input("Sepal length", value=5.1)
sepal_width = st.number_input("Sepal width", value=3.5)
petal_length = st.number_input("Petal length", value=1.4)
petal_width = st.number_input("Petal width", value=0.2)

if st.button("Predict"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    try:
        res = requests.post("http://localhost:8000/predict", json=data)
        result = res.json()
        st.write("Prediction:", result["class_name"])
    except:
        st.write("Error: API is not running")