import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Set page configuration
st.set_page_config(page_title="Mobile Price Classification", page_icon=":iphone:", layout="centered")

# Display an image
st.image(r"Screenshot 2024-07-25 185502.png")
st.image(r"Mobile.jpeg")

# Title and header
st.title("Mobile Price Classification Project")
st.subheader("By Shravan Kumar Polu")
st.markdown("Predict the price range of a mobile based on its specifications.")

# Load the model
model = pickle.load(open(r"MobilePriceClassification.pkl", "rb"))

# Define a function to map the dual SIM input
def map_dual_sim(value):
    return "Yes" if value == 1 else "No"

# Input fields
battery_power = st.slider("Battery Power (mAh)", 500, 5000, step=100)
dual_sim = st.radio("Dual SIM", [1, 0], format_func=map_dual_sim)
fc = st.slider("Front Camera (MP)", 0, 20, step=1)
four_g = st.radio("4G Supported", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
int_memory = st.slider("Internal Memory (GB)", 2, 256, step=2)
mobile_wt = st.slider("Mobile Weight (g)", 80, 250, step=10)
n_cores = st.slider("Number of Cores", 1, 8, step=1)
pc = st.slider("Primary Camera (MP)", 0, 48, step=1)
ram = st.slider("RAM (MB)", 256, 8000, step=256)
sc_h = st.slider("Screen Height (cm)", 5, 20, step=1)
sc_w = st.slider("Screen Width (cm)", 5, 20, step=1)

# Submit button
if st.button("Submit"):
    features = np.array([[battery_power, dual_sim, fc, four_g, int_memory, mobile_wt, n_cores, pc, ram, sc_h, sc_w]])
    price_range = model.predict(features)[0]
    
    # Price range description
    price_range_desc = ["Low cost (under ₹10,000)", "Mid cost (10,000 - ₹25,000)", "High cost (₹25,000 - ₹50,000)", "Very high cost (Above ₹50,000)"][price_range]
    st.success(f"The Estimated Price Range is: **{price_range_desc}**")

# Footer
st.markdown("---")
st.markdown("© 2024 Shravan Kumar Polu")
