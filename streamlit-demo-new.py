import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Title
st.title("Health Insurance Plan Affordability Predictor")

# Sidebar user input
st.sidebar.header("Input Plan Features")

def user_input_features():
    CopayInnTier2 = st.sidebar.slider('Copay In-Network Tier 2 ($)', 0, 500, 50)
    CopayOutofNet = st.sidebar.slider('Copay Out-of-Network ($)', 0, 2500, 500)
    CoinsInnTier1 = st.sidebar.slider('Coinsurance In-Network Tier 1 (%)', 0, 100, 20)
    CoinsInnTier2 = st.sidebar.slider('Coinsurance In-Network Tier 2 (%)', 0, 100, 20)
    CoinsOutofNet = st.sidebar.slider('Coinsurance Out-of-Network (%)', 0, 100, 50)
    LimitQty = st.sidebar.slider('Service Limit Quantity', 0, 50, 5)
    IsExclFromInnMOOP = st.sidebar.selectbox('Excluded from In-Network MOOP?', [0, 1])
    IsExclFromOonMOOP = st.sidebar.selectbox('Excluded from Out-of-Network MOOP?', [0, 1])
    IsCovered = st.sidebar.selectbox('Benefit Covered?', [0, 1])
    IsEHB = st.sidebar.selectbox('Essential Health Benefit?', [0, 1])

    data = {
        'CopayInnTier2': CopayInnTier2,
        'CopayOutofNet': CopayOutofNet,
        'CoinsInnTier1': CoinsInnTier1,
        'CoinsInnTier2': CoinsInnTier2,
        'CoinsOutofNet': CoinsOutofNet,
        'LimitQty': LimitQty,
        'IsExclFromInnMOOP': IsExclFromInnMOOP,
        'IsExclFromOonMOOP': IsExclFromOonMOOP,
        'IsCovered': IsCovered,
        'IsEHB': IsEHB
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Pretrained model setup (mock model here for demo)
# In real usage, you would load a model: rf_model = joblib.load("model.pkl")
model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
scaler = StandardScaler()

# Dummy training on small data to simulate prediction
train_data = pd.DataFrame({
    'CopayInnTier2': [10, 20, 30],
    'CopayOutofNet': [100, 200, 300],
    'CoinsInnTier1': [10, 20, 30],
    'CoinsInnTier2': [5, 10, 15],
    'CoinsOutofNet': [50, 60, 70],
    'LimitQty': [1, 2, 3],
    'IsExclFromInnMOOP': [0, 1, 0],
    'IsExclFromOonMOOP': [1, 0, 0],
    'IsCovered': [1, 0, 1],
    'IsEHB': [1, 1, 0]
})
train_target = [40, 70, 55]  # Simulated affordability scores
train_scaled = scaler.fit_transform(train_data)
model.fit(train_scaled, train_target)

# Prediction
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

# Display prediction
st.subheader("Predicted Affordability Score")
st.success(f"Estimated Score: {prediction[0]:.2f}")
