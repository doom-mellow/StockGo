import joblib
import pandas as pd


# ==========================================================
# FEATURE COLUMNS
# ==========================================================

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Return_5D",
    "Price_Change",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "Volatility_20",
    "Bollinger_Middle",
    "Bollinger_Upper",
    "Bollinger_Lower",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volume_Change"
]


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(
    model_path="models/saved_models/xgboost_multi_stock.pkl"
):

    model = joblib.load(model_path)

    return model


# ==========================================================
# PREDICT NEXT DAY
# ==========================================================

def predict_next_day(
    data: pd.DataFrame,
    model
):

    data = data.copy()

    # ------------------------------------------------------
    # Convert date
    # ------------------------------------------------------

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    # ------------------------------------------------------
    # Sort by date
    # ------------------------------------------------------

    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Convert model features to numeric
    # ------------------------------------------------------

    for column in FEATURE_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Remove rows with invalid feature values
    # ------------------------------------------------------

    data = data.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Get latest observation
    # ------------------------------------------------------

    latest = data.iloc[-1]

    # ------------------------------------------------------
    # Current stock price
    # ------------------------------------------------------

    current_price = float(
        latest["Close"]
    )

    # ------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------

    X_latest = (
        latest[FEATURE_COLUMNS]
        .astype(float)
        .to_frame()
        .T
    )

    # ------------------------------------------------------
    # Predict next-day return
    # ------------------------------------------------------

    predicted_return = float(
        model.predict(X_latest)[0]
    )

    # ------------------------------------------------------
    # Convert predicted return to price
    # ------------------------------------------------------

    predicted_price = (
        current_price
        * (1 + predicted_return)
    )

    # ------------------------------------------------------
    # Determine direction
    # ------------------------------------------------------

    if predicted_return > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    # ------------------------------------------------------
    # Return prediction
    # ------------------------------------------------------

    return {
        "current_price": current_price,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "direction": direction
    }