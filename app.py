import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown("""
<style>

/* Full Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* DARK OVERLAY */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.70);
    z-index: 0;
}

/* CONTENT */
.block-container {
    position: relative;
    z-index: 1;
}

/* Transparent main container */
.block-container {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin:50px;
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff;
    font-size: 40px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
}

/* Text */
.stMarkdown p {
    color: #f0f8ff;
    text-align: center;
    font-size: 18px;
}

/* Labels */
label {
    color: #000000 !important;
    font-weight: 600;
}

/* Input fields */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #36d1dc, #5b86e5);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #5b86e5, #36d1dc);
}

/* Result box */
.stSuccess {
    background-color: rgba(0, 255, 200, 0.25);
    border-radius: 12px;
    padding: 15px;
    font-size: 18px;
    color: #003333;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

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