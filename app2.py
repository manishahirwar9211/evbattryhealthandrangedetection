import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="EV Battery System", layout="wide")

# ------------------------------
# Title
# ------------------------------
st.title("🚗 EV Battery Monitoring Dashboard")
st.markdown("### 🔋 AI-Based Range & Battery Health Prediction")

# ------------------------------
# Dataset
# ------------------------------
data = {
    "Voltage": [48,47.8,47.5,47.2,47,46.5,46,45.5,45,44.5,
                44,43.5,43,42.5,42,41.5,41,40.5,40,39.5],

    "Current": [10,11,12,13,14,15,16,17,18,19,
                20,21,22,23,24,25,26,27,28,29],

    "Temperature": [30,31,32,34,35,36,38,39,40,41,
                    42,43,45,46,48,49,50,52,53,55],

    "Speed": [40,42,45,48,50,52,55,58,60,62,
              65,68,70,72,75,78,80,82,85,88],

    "Load": [70,72,75,78,80,82,85,88,90,92,
             95,98,100,105,110,115,120,125,130,135],

    "Cycles": [100,102,105,108,110,115,120,125,130,135,
               140,145,150,155,160,165,170,175,180,185],

    "SoH": [90,89,88,87,85,84,82,81,79,77,
            75,73,70,68,65,63,60,58,55,52],

    "Range": [120,118,115,112,110,105,100,95,90,85,
              80,75,70,65,60,55,50,45,40,35]
}

df = pd.DataFrame(data)

# ------------------------------
# Train Models
# ------------------------------
X_range = df[["Voltage","Current","Temperature","Speed","Load","Cycles","SoH"]]
y_range = df["Range"]

range_model = RandomForestRegressor(n_estimators=100, random_state=42)
range_model.fit(X_range, y_range)

X_soh = df[["Voltage","Current","Temperature","Speed","Load","Cycles"]]
y_soh = df["SoH"]

soh_model = RandomForestRegressor(n_estimators=100, random_state=42)
soh_model.fit(X_soh, y_soh)

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("⚙️ Battery Parameters")

voltage = st.sidebar.slider("Voltage (V)", 35.0, 50.0, 45.0)
current = st.sidebar.slider("Current (A)", 5.0, 30.0, 15.0)
temperature = st.sidebar.slider("Temperature (°C)", 20.0, 60.0, 35.0)
speed = st.sidebar.slider("Speed (km/h)", 20.0, 100.0, 50.0)
load = st.sidebar.slider("Load (%)", 50.0, 150.0, 80.0)
cycles = st.sidebar.slider("Charge Cycles", 50, 200, 120)

# ------------------------------
# Prediction
# ------------------------------
soh_input = [[voltage, current, temperature, speed, load, cycles]]
predicted_soh = soh_model.predict(soh_input)[0]

range_input = [[voltage, current, temperature, speed, load, cycles, predicted_soh]]
predicted_range = range_model.predict(range_input)[0]

# ------------------------------
# Output Section
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("🔋 Battery Health (SoH)", f"{predicted_soh:.2f} %")

with col2:
    st.metric("🚗 Predicted Range", f"{predicted_range:.2f} km")

# ------------------------------
# Alerts
# ------------------------------
st.subheader("⚠️ Alerts")

if temperature > 45:
    st.error("🔥 High Temperature Detected!")

elif predicted_soh < 60:
    st.warning("⚠️ Battery Health is Low")

else:
    st.success("✅ Battery is in Good Condition")

# ------------------------------
# Dynamic Visualization
# ------------------------------
st.subheader("📊 Real-Time Input Visualization")

input_df = pd.DataFrame({
    "Parameter": ["Voltage","Temperature","Load","SoH"],
    "Value": [voltage, temperature, load, predicted_soh]
})

st.bar_chart(input_df.set_index("Parameter"))

# ------------------------------
# Historical Trends
# ------------------------------
st.subheader("📈 Historical Trends")

col3, col4 = st.columns(2)

with col3:
    st.line_chart(df["Voltage"])

with col4:
    st.line_chart(df["Range"])

# ------------------------------
# Dataset
# ------------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df)
