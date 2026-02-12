import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------
# Load Trained Model
# ---------------------------
model = joblib.load("phone_price_model.pkl")

st.set_page_config(page_title="Mobile Price Predictor", layout="wide")

st.title("📱 Mobile Price Prediction App")
st.write("Predict mobile price using Linear Regression Pipeline")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter Mobile Specifications")

# Categorical Inputs
brand = st.sidebar.selectbox(
    "Brand",
    ["Samsung", "Apple", "Xiaomi", "Realme", "Oneplus","Motorola"]
)

processor_brand = st.sidebar.selectbox(
    "Processor Brand",
    ["Snapdragon", "Dimensity", "Exynos", "Bionic"]
)

os = st.sidebar.selectbox(
    "Operating System",
    ["Android", "iOS"]
)

# Numerical Inputs
reting = st.sidebar.slider("Rating", 1.0, 5.0, 4.0)
is_5g = st.sidebar.selectbox("5G Support", [0, 1])
core = st.sidebar.number_input("Number of Cores", 1, 12, 8)
is_nfc = st.sidebar.selectbox("NFC Support", [0, 1])
proccessor_speed = st.sidebar.number_input("Processor Speed (GHz)", 0.5, 5.0, 2.5)
ram = st.sidebar.number_input("RAM (GB)", 1, 24, 8)
internal_memory = st.sidebar.number_input("Internal Memory (GB)", 8, 1024, 128)
rear_mp = st.sidebar.number_input("Rear Camera (MP)", 2, 200, 64)
front_mp = st.sidebar.number_input("Front Camera (MP)", 2, 100, 16)
battery_size = st.sidebar.number_input("Battery Size (mAh)", 1000, 10000, 5000)
display_size = st.sidebar.number_input("Display Size (inch)", 4.0, 8.0, 6.5)
refresh_rate = st.sidebar.number_input("Refresh Rate (Hz)", 60, 240, 120)
charging_speed = st.sidebar.number_input("Charging Speed", 45,90,60)
is_ir_blaster = st.sidebar.selectbox("Ir Blast",[0,1])
fast_charge = st.sidebar.selectbox("fast_charge",[0,1])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Price 💰"):

    # Create DataFrame
    input_data = pd.DataFrame([{
        "brand": brand,
        "processor_brand": processor_brand,
        "os": os,
        "reting": reting,
        "is_5g": is_5g,
        "core": core,
        "is_nfc": is_nfc,
        "proccessor_speed": proccessor_speed,
        "ram": ram,
        "internal_memory": internal_memory,
        "rear_mp": rear_mp,
        "front_mp": front_mp,
        "battery_size": battery_size,
        "display_size": display_size,
        "refresh_rate": refresh_rate,
        "charging_speed": charging_speed,
        "is_ir_blaster": is_ir_blaster,
        "fast_charge": fast_charge

        
    }])

    # Log Prediction
    log_price = model.predict(input_data)[0]

    # Convert Back from log1p
    predicted_price = np.expm1(log_price)

    st.success(f"💰 Predicted Price: ₹ {predicted_price:,.2f}")