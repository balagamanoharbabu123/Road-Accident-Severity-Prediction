import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Road Accident Severity Predictor", page_icon="🚦")

# Load the trained model and the dropdown options saved by the notebook
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_options():
    with open("options.json") as f:
        return json.load(f)

model = load_model()
options = load_options()

st.title("🚦 Road Accident Severity Predictor")
st.write(
    "Enter the conditions of a road accident to see which severity level is most likely. "
    "Built with a Random Forest model trained on Indian road accident records."
)

col1, col2 = st.columns(2)

with col1:
    hour = st.selectbox("Hour of the day (0-23)", options["Hour"], index=12)
    day = st.selectbox("Day of week", options["Day_of_week"])
    age = st.selectbox("Driver age band", options["Age_band_of_driver"])
    experience = st.selectbox("Driving experience", options["Driving_experience"])
    vehicle = st.selectbox("Type of vehicle", options["Type_of_vehicle"])
    n_vehicles = st.selectbox("Number of vehicles involved", options["Number_of_vehicles_involved"])

with col2:
    area = st.selectbox("Area of accident", options["Area_accident_occured"])
    junction = st.selectbox("Type of junction", options["Types_of_Junction"])
    surface = st.selectbox("Road surface condition", options["Road_surface_conditions"])
    light = st.selectbox("Light conditions", options["Light_conditions"])
    weather = st.selectbox("Weather", options["Weather_conditions"])
    cause = st.selectbox("Cause of accident", options["Cause_of_accident"])

if st.button("Predict severity"):
    # Column names must match exactly what the model was trained on
    row = pd.DataFrame([{
        "Day_of_week": day,
        "Age_band_of_driver": age,
        "Driving_experience": experience,
        "Type_of_vehicle": vehicle,
        "Area_accident_occured": area,
        "Types_of_Junction": junction,
        "Road_surface_conditions": surface,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Cause_of_accident": cause,
        "Hour": hour,
        "Number_of_vehicles_involved": n_vehicles,
    }])

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]

    st.subheader(f"Predicted severity: {prediction}")

    prob_table = pd.DataFrame({
        "Severity": model.classes_,
        "Probability (%)": (probabilities * 100).round(1),
    }).set_index("Severity")
    st.bar_chart(prob_table)
    st.table(prob_table)

    st.caption(
        "The model is trained with class weights to catch severe cases, so it raises many false alarms "
        "for Serious and Fatal. This is an educational project, not a safety tool."
    )
