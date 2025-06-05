import os
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import requests
from textblob import TextBlob

# Use Streamlit secrets in deployment
API_KEY = st.secrets.get("BINANCE_API_KEY", "")
API_SECRET = st.secrets.get("BINANCE_API_SECRET", "")
CRYPTOPANIC_KEY = st.secrets.get("CRYPTOPANIC_API_KEY", "")

st.set_page_config(layout="wide")
st.title("📊 Multi-Coin LSTM Crypto Predictor + News Sentiment")
coins = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SHIBA": "SHIB-USD",
    "PEPE": "PEPE-USD"
}

@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="60d", interval="1h")
    return df

def get_news():
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_KEY}&public=true"
    response = requests.get(url)
    if response.status_code == 200:
        return [item["title"] for item in response.json().get("results", [])[:5]]
    return []

def get_sentiment_score(headlines):
    score = 0
    for title in headlines:
        score += TextBlob(title).sentiment.polarity
    return score / len(headlines) if headlines else 0

for name, symbol in coins.items():
    st.subheader(f"🪙 {name}")
    df = get_data(symbol)

    data = df[["Close"]]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    seq_len = 50
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=3, batch_size=16, verbose=0)

    last_seq = scaled_data[-seq_len:]
    last_seq = last_seq.reshape(1, seq_len, 1)
    next_scaled = model.predict(last_seq)
    next_price = scaler.inverse_transform(next_scaled)[0][0]
    current_price = data["Close"].iloc[-1]
    pct_change = ((next_price - current_price) / current_price) * 100

    # 🧠 Sentiment
    news = get_news()
    sentiment = get_sentiment_score(news)
    adjusted_pct = pct_change + (sentiment * 2)
    signal = "BUY" if adjusted_pct > 0.5 else "SELL" if adjusted_pct < -0.5 else "HOLD"

    # 📈 Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=[data.index[-1], data.index[-1] + pd.Timedelta(hours=1)],
                             y=[current_price, next_price],
                             mode="lines+markers", name="Prediction"))
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Metrics
    st.metric("Current Price", f"${current_price:,.4f}")
    st.metric("Predicted Price", f"${next_price:,.4f}", delta=f"{pct_change:.2f}%")
    st.markdown(f"### 📌 Signal: `{signal}`", unsafe_allow_html=True)

    # 📰 News + Sentiment
    st.markdown(f"🧠 Sentiment Score: `{sentiment:.2f}`")
    for n in news:
        st.caption(f"🗞️ {n}")

    # 📥 Download
    csv = df.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv, file_name=f"{symbol}_data.csv", mime='text/csv')

st.caption("🚀 Built with Streamlit, LSTM, Binance API, and ❤️")
streamlit
yfinance
numpy
pandas
plotly
keras
tensorflow
requests
textblob
scikit-learn
python-dotenv
BINANCE_API_KEY = "DzISiDAIXDXhlTWexvdkVV68liVjMQg21qNTCq1GmyKyoKOBzuB86x7dDAZ4F2pr"
BINANCE_API_SECRET = "taNebZq7Wo7IL7QIudjs32m8EqAWtKtPvgLnUxFzoqmRcQb5dNDC1x0SbwWfQZGl"
CRYPTOPANIC_API_KEY = "6b423399480ec7dcfc8e96ff5ca1d85e3b125a4b"
