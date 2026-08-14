import pandas as pd
from pathlib import Path


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical and statistical features
    from historical stock market data.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned stock market data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing engineered features.
    """

    data = data.copy()

    # ==========================================================
    # 1. PRICE-BASED FEATURES
    # ==========================================================

    # Daily percentage return
    data["Daily_Return"] = data["Close"].pct_change()

    # Five-day percentage return
    data["Return_5D"] = data["Close"].pct_change(periods=5)

    # Daily price change
    data["Price_Change"] = data["Close"] - data["Open"]


    # ==========================================================
    # 2. MOVING AVERAGES
    # ==========================================================

    # 20-day Simple Moving Average
    data["SMA_20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    # 50-day Simple Moving Average
    data["SMA_50"] = (
        data["Close"]
        .rolling(window=50)
        .mean()
    )

    # 20-day Exponential Moving Average
    data["EMA_20"] = (
        data["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )


    # ==========================================================
    # 3. VOLATILITY
    # ==========================================================

    # 20-day rolling volatility
    data["Volatility_20"] = (
        data["Daily_Return"]
        .rolling(window=20)
        .std()
    )


    # ==========================================================
    # 4. BOLLINGER BANDS
    # ==========================================================

    rolling_mean = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    rolling_std = (
        data["Close"]
        .rolling(window=20)
        .std()
    )

    data["Bollinger_Middle"] = rolling_mean

    data["Bollinger_Upper"] = (
        rolling_mean + (2 * rolling_std)
    )

    data["Bollinger_Lower"] = (
        rolling_mean - (2 * rolling_std)
    )


    # ==========================================================
    # 5. RSI
    # ==========================================================

    price_change = data["Close"].diff()

    gains = price_change.clip(lower=0)

    losses = -price_change.clip(upper=0)

    average_gain = (
        gains
        .rolling(window=14)
        .mean()
    )

    average_loss = (
        losses
        .rolling(window=14)
        .mean()
    )

    relative_strength = (
        average_gain / average_loss
    )

    data["RSI_14"] = (
        100 - (100 / (1 + relative_strength))
    )


    # ==========================================================
    # 6. MACD
    # ==========================================================

    ema_12 = (
        data["Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema_26 = (
        data["Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    data["MACD"] = ema_12 - ema_26

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["MACD_Histogram"] = (
        data["MACD"] - data["MACD_Signal"]
    )


    # ==========================================================
    # 7. VOLUME FEATURES
    # ==========================================================

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )


    # ==========================================================
    # 8. PREDICTION TARGET
    # ==========================================================

    # Tomorrow's percentage return
    data["Target"] = (
        data["Close"].shift(-1) / data["Close"] - 1
    )


    # ==========================================================
    # 9. CLEAN FEATURE DATA
    # ==========================================================

    # Remove rows containing NaN values generated
    # by rolling calculations and target shifting.
    data = data.dropna()

    # Reset index
    data = data.reset_index(drop=True)

    return data


def save_feature_data(
    data: pd.DataFrame,
    ticker: str,
    period: str
):
    """
    Save feature-engineered stock data.
    """

    processed_directory = Path("data/processed")

    processed_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = f"{ticker}_{period}_features.csv"

    file_path = processed_directory / filename

    data.to_csv(
        file_path,
        index=False
    )

    print(
        f"Feature data saved to: {file_path}"
    )