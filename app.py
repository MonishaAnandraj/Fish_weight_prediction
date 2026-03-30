import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🐟 Fish Weight Predictor", layout="centered")

st.title("🐟 Fish Weight Prediction App")
st.write("Enter fish measurements to predict weight")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------------- USER INPUTS ----------------
species = st.selectbox("Fish Species", [
    "Bream", "Roach", "Whitefish", "Parkki",
    "Perch", "Pike", "Smelt"
])

length1 = st.number_input("Length (cm)", min_value=0.0, value=20.0)
height = st.number_input("Height (cm)", min_value=0.0, value=10.0)
width = st.number_input("Width (cm)", min_value=0.0, value=5.0)
girth = st.number_input("Girth (cm)", min_value=0.0, value=15.0)

# ---------------- FEATURE ENGINEERING ----------------
input_data = pd.DataFrame({
    "length1_cm": [length1],
    "height_cm": [height],
    "width_cm": [width],
    "girth_cm": [girth]
})

# Derived features (same as notebook)
input_data["volume_proxy"] = input_data["length1_cm"] * input_data["height_cm"] * input_data["width_cm"]
input_data["log_volume"] = np.log(input_data["volume_proxy"] + 1e-6)
input_data["length_sq"] = input_data["length1_cm"] ** 2
input_data["length_cu"] = input_data["length1_cm"] ** 3
input_data["length_x_girth"] = input_data["length1_cm"] * input_data["girth_cm"]

# Dummy encoding for species
species_df = pd.get_dummies(pd.DataFrame({"species": [species]}), drop_first=True)

# Merge input + species
input_data = pd.concat([input_data, species_df], axis=1)

# ---------------- COLUMN ALIGNMENT ----------------
# Ensure all columns exist
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[model_columns]

# ---------------- PREDICTION ----------------
if st.button("Predict Weight"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Fish Weight: {prediction:.2f} grams")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit | Machine Learning Model")