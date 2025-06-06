import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Crypto Price Prediction Dashboard", layout="wide")

# Load pre-trained model if available
def load_lstm_model():
    try:
        model = load_model("btc_model.h5")
        return model
    except Exception as e:
        st.warning(f"Error loading model: {e}")
        return None

# Fetch live price from Binance API
def fetch_live_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        st.error(f"Error fetching {symbol} price: {e}")
        return None

# Generate dummy historical data
def generate_dummy_data():
    now = datetime.datetime.now()
    dates = pd.date_range(end=now, periods=100)
    prices = np.cumsum(np.random.randn(100)) + 27000
    df = pd.DataFrame({"Date": dates, "Close": prices})
    return df

# Preprocess data for prediction
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    X_test = [scaled_data[-60:]]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test, scaler

# Predict next price
def predict_price(model, df):
    if model is None:
        return -0.02
    X_test, scaler = preprocess_data(df)
    prediction = model.predict(X_test)
    return scaler.inverse_transform(prediction)[0][0]

# Plot historical data
def plot_chart(df, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Historical Close Price")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USDT)")
    ax.legend()
    st.pyplot(fig)

# Sidebar
st.sidebar.title("🧠 Crypto Predictor")
selected_coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "PEPEUSDT", "SHIBUSDT"])
st.sidebar.markdown("---")

# Main
st.title("📊 Crypto Price Prediction Dashboard")
live_price = fetch_live_price(selected_coin)
model = load_lstm_model()
data = generate_dummy_data()
predicted_price = predict_price(model, data)

# Display metrics
col1, col2 = st.columns(2)
with col1:
    if live_price:
        st.metric(label=f"Live {selected_coin[:-4]} Price", value=f"${live_price:,.2f}")
    else:
        st.warning("Live price unavailable.")

with col2:
    st.metric(label=f"Predicted {selected_coin[:-4]} Price (Next Step)", value=f"${predicted_price:,.2f}")

plot_chart(data, f"{selected_coin[:-4]} Historical Price")



