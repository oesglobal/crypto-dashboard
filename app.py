import streamlit as st
import requests
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# --- Get Live Price from Binance ---
def get_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['price']) if 'price' in data else None
    except Exception as e:
        print("Error fetching price:", e)
        return None

# --- Predict Price Using LSTM Model ---
def predict_price(model, scaler, price):
    try:
        scaled = scaler.transform(np.array([[price]]))
        seq = np.reshape(scaled, (1, 1, 1))  # 1 sequence, 1 step, 1 feature
        pred_scaled = model.predict(seq)
        pred = scaler.inverse_transform(pred_scaled)
        return float(pred[0][0])
    except Exception as e:
        print("Prediction error:", e)
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Crypto Predictor", layout="centered")
st.title("📊 Crypto Price Prediction Dashboard")

# --- Load Model ---
model_path = "btc_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    scaler = MinMaxScaler()
    scaler.fit(np.array([[10000], [100000]]))  # Assume BTC range for scaling
else:
    model = None
    st.error("❌ Model file `btc_model.h5` not found. Please upload it.")
    st.stop()

# --- Get Live BTC Price ---
live_price = get_price("BTCUSDT")
if live_price:
    st.metric("🔴 Live BTC Price", f"${live_price:,.2f}")
else:
    st.warning("⚠️ Could not fetch live BTC price.")
    st.stop()

# --- Prediction ---
predicted_price = predict_price(model, scaler, live_price)
if predicted_price:
    st.metric("📈 Predicted BTC Price (Next Step)", f"${predicted_price:,.2f}")
else:
    st.warning("⚠️ Prediction failed.")
