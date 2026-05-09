import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

# ─── Sidebar Inputs ────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# ─── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    return df

if st.sidebar.button("🚀 Run Prediction"):
    df = load_data(ticker, start, end)

    # ─── Candlestick Chart ─────────────────────────────────
    st.subheader(f"📊 {ticker} Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ─── Moving Averages ───────────────────────────────────
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    st.subheader("📉 Moving Averages")
    st.line_chart(df[['Close', 'MA_20', 'MA_50']])

    # ─── RSI ───────────────────────────────────────────────
    def compute_rsi(data, window=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = compute_rsi(df['Close'])
    st.subheader("📈 RSI Indicator")
    st.line_chart(df[['RSI']])

    # ─── Linear Regression Prediction ──────────────────────
    st.subheader("🤖 Next Day Price Prediction")

    from sklearn.linear_model import LinearRegression

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df.dropna(inplace=True)

    features = ['MA_20', 'MA_50', 'RSI', 'MACD', 'Volume']
    X = df[features]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_row = df[features].iloc[-1].values.reshape(1, -1)
    pred_price = model.predict(last_row)[0]

    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    col2.metric("Predicted Next Day", f"${pred_price:.2f}")

    # ─── Prediction Chart ──────────────────────────────────
    st.subheader("📊 Actual vs Predicted Prices")
    df['Predicted'] = model.predict(X)
    st.line_chart(df[['Close', 'Predicted']])