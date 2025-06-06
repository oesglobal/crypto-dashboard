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
    import streamlit as st
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

st.set_page_config(page_title="Crypto LSTM Dashboard", layout="wide")
st.title("Crypto Price Prediction Dashboard")

# Cache model loading so it doesn't reload on every rerun
@st.cache_resource
def load_lstm_model():
    # If you have a saved model file, load it here:
    # return tf.keras.models.load_model("model_btc_lstm.h5")

    # Otherwise, build a simple example LSTM model
    model = Sequential([
        tf.keras.layers.Input(shape=(60, 1)),  # 60 timesteps, 1 feature
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

model = load_lstm_model()

# Fetch live BTC price from Binance API
def get_btc_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        st.error(f"Error fetching BTC price: {e}")
        return None

btc_price = get_btc_price()
if btc_price is not None:
    st.write(f"Current BTC Price (USDT): ${btc_price:,.2f}")

# Prepare dummy input data for prediction (replace with your real processed data)
def prepare_dummy_input():
    # 1 sample, 60 timesteps, 1 feature with random data
    return np.random.rand(1, 60, 1).astype(np.float32)

input_data = prepare_dummy_input()

# Predict price using the model (dummy prediction here)
prediction = model.predict(input_data)
predicted_price = prediction.flatten()[0]

st.write(f"Predicted BTC Price (dummy): ${predicted_price:.2f}")

st.info("Replace dummy input and model with your actual data and trained model.")
# Install dependencies (only once)

