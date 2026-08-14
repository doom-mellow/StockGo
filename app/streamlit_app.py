import sys
from pathlib import Path

import streamlit as st
import pandas as pd


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from src.models.predict import (
    load_model,
    predict_next_day
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="StockGo",
    page_icon="📈",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("📈 StockGo")

st.subheader(
    "AI-Powered Stock Return Prediction"
)

st.write(
    "StockGo uses machine learning and technical indicators "
    "to estimate the next trading day's stock return and price."
)


# ==========================================================
# FILE PATHS
# ==========================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_stock_2y_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "xgboost_multi_stock.pkl"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    return data


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_stock_model():

    return load_model(
        str(MODEL_PATH)
    )


data = load_data()

model = load_stock_model()


# ==========================================================
# STOCK SELECTION
# ==========================================================

st.divider()

st.subheader("Select a Stock")

available_stocks = sorted(
    data["Ticker"].unique()
)

selected_stock = st.selectbox(
    "Choose a stock to analyze:",
    available_stocks
)


# ==========================================================
# FILTER SELECTED STOCK
# ==========================================================

stock_data = data[
    data["Ticker"] == selected_stock
].copy()

stock_data = stock_data.sort_values(
    "Date"
).reset_index(drop=True)


# ==========================================================
# MAKE PREDICTION
# ==========================================================

result = predict_next_day(
    stock_data,
    model
)


current_price = result[
    "current_price"
]

predicted_return = result[
    "predicted_return"
]

predicted_price = result[
    "predicted_price"
]

direction = result[
    "direction"
]


# ==========================================================
# PREDICTION HEADER
# ==========================================================

st.divider()

st.subheader(
    f"📊 {selected_stock} Prediction"
)


# ==========================================================
# DISPLAY METRICS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Price",
        f"${current_price:.2f}"
    )


with col2:

    price_change = (
        predicted_price
        - current_price
    )

    st.metric(
        "Predicted Price",
        f"${predicted_price:.2f}",
        f"{price_change:+.2f}"
    )


with col3:

    st.metric(
        "Expected Return",
        f"{predicted_return * 100:.3f}%"
    )


with col4:

    st.metric(
        "Direction",
        direction
    )


# ==========================================================
# PREDICTION MESSAGE
# ==========================================================

st.divider()

if direction == "UP":

    st.success(
        f"📈 StockGo predicts that {selected_stock} "
        f"could move UP by approximately "
        f"{predicted_return * 100:.3f}% "
        f"on the next trading day."
    )

else:

    st.error(
        f"📉 StockGo predicts that {selected_stock} "
        f"could move DOWN by approximately "
        f"{abs(predicted_return) * 100:.3f}% "
        f"on the next trading day."
    )


# ==========================================================
# RECENT PRICE HISTORY
# ==========================================================

st.divider()

st.subheader(
    f"📈 {selected_stock} — Recent Price History"
)


recent_data = stock_data[
    ["Date", "Close"]
].tail(30)


recent_data = recent_data.set_index(
    "Date"
)


st.line_chart(
    recent_data["Close"]
)


# ==========================================================
# TECHNICAL INDICATORS
# ==========================================================

st.divider()

st.subheader(
    "Technical Indicators"
)


indicator_col1, indicator_col2, indicator_col3, indicator_col4 = (
    st.columns(4)
)


latest = stock_data.iloc[-1]


with indicator_col1:

    st.metric(
        "RSI (14)",
        f"{latest['RSI_14']:.2f}"
    )


with indicator_col2:

    st.metric(
        "SMA (20)",
        f"${latest['SMA_20']:.2f}"
    )


with indicator_col3:

    st.metric(
        "SMA (50)",
        f"${latest['SMA_50']:.2f}"
    )


with indicator_col4:

    st.metric(
        "Volatility (20D)",
        f"{latest['Volatility_20'] * 100:.2f}%"
    )


# ==========================================================
# RECENT MARKET DATA
# ==========================================================

st.divider()

st.subheader(
    "Recent Market Data"
)


display_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


recent_market_data = stock_data[
    display_columns
].tail(10).copy()


st.dataframe(
    recent_market_data,
    width="stretch"
)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)

st.write(
    "StockGo uses a tuned XGBoost regression model "
    "trained on historical data from multiple stocks."
)

st.write(
    "The model predicts the next trading day's "
    "percentage return using price, volume, momentum, "
    "volatility, moving averages, Bollinger Bands, RSI "
    "and MACD-based features."
)


# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()

st.caption(
    "⚠️ StockGo is an educational machine-learning project. "
    "Predictions are estimates based on historical market data "
    "and should not be considered financial advice."
)