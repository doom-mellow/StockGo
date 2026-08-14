import pandas as pd
from pathlib import Path


def create_multi_stock_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create technical and statistical features
    separately for each stock.
    """

    data = data.copy()

    # Make sure data is sorted correctly
    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    feature_groups = []

    # ==========================================================
    # PROCESS EACH STOCK SEPARATELY
    # ==========================================================

    for ticker, stock_data in data.groupby(
        "Ticker",
        group_keys=False
    ):

        stock_data = stock_data.copy()

        # ======================================================
        # 1. PRICE-BASED FEATURES
        # ======================================================

        stock_data["Daily_Return"] = (
            stock_data["Close"].pct_change()
        )

        stock_data["Return_5D"] = (
            stock_data["Close"].pct_change(
                periods=5
            )
        )

        stock_data["Price_Change"] = (
            stock_data["Close"]
            - stock_data["Open"]
        )

        # ======================================================
        # 2. MOVING AVERAGES
        # ======================================================

        stock_data["SMA_20"] = (
            stock_data["Close"]
            .rolling(window=20)
            .mean()
        )

        stock_data["SMA_50"] = (
            stock_data["Close"]
            .rolling(window=50)
            .mean()
        )

        stock_data["EMA_20"] = (
            stock_data["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
        )

        # ======================================================
        # 3. VOLATILITY
        # ======================================================

        stock_data["Volatility_20"] = (
            stock_data["Daily_Return"]
            .rolling(window=20)
            .std()
        )

        # ======================================================
        # 4. BOLLINGER BANDS
        # ======================================================

        rolling_mean = (
            stock_data["Close"]
            .rolling(window=20)
            .mean()
        )

        rolling_std = (
            stock_data["Close"]
            .rolling(window=20)
            .std()
        )

        stock_data["Bollinger_Middle"] = (
            rolling_mean
        )

        stock_data["Bollinger_Upper"] = (
            rolling_mean
            + (2 * rolling_std)
        )

        stock_data["Bollinger_Lower"] = (
            rolling_mean
            - (2 * rolling_std)
        )

        # ======================================================
        # 5. RSI
        # ======================================================

        price_change = (
            stock_data["Close"].diff()
        )

        gains = price_change.clip(
            lower=0
        )

        losses = -price_change.clip(
            upper=0
        )

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

        stock_data["RSI_14"] = (
            100
            - (
                100
                / (1 + relative_strength)
            )
        )

        # ======================================================
        # 6. MACD
        # ======================================================

        ema_12 = (
            stock_data["Close"]
            .ewm(
                span=12,
                adjust=False
            )
            .mean()
        )

        ema_26 = (
            stock_data["Close"]
            .ewm(
                span=26,
                adjust=False
            )
            .mean()
        )

        stock_data["MACD"] = (
            ema_12 - ema_26
        )

        stock_data["MACD_Signal"] = (
            stock_data["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        stock_data["MACD_Histogram"] = (
            stock_data["MACD"]
            - stock_data["MACD_Signal"]
        )

        # ======================================================
        # 7. VOLUME FEATURES
        # ======================================================

        stock_data["Volume_Change"] = (
            stock_data["Volume"].pct_change()
        )

        # ======================================================
        # 8. PREDICTION TARGET
        # ======================================================

        # Tomorrow's percentage return
        stock_data["Target"] = (
            stock_data["Close"]
            .shift(-1)
            / stock_data["Close"]
            - 1
        )

        feature_groups.append(
            stock_data
        )

    # ==========================================================
    # COMBINE ALL STOCKS
    # ==========================================================

    result = pd.concat(
        feature_groups,
        ignore_index=True
    )

    # Remove rows created by rolling calculations
    # and final target shift.
    result = result.dropna()

    result = result.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    return result


def save_multi_stock_features(
    data: pd.DataFrame,
    filename: str = "multi_stock_2y_features.csv"
):

    directory = Path(
        "data/processed"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        directory / filename
    )

    data.to_csv(
        file_path,
        index=False
    )

    print(
        f"Feature data saved to: "
        f"{file_path}"
    )

    return file_path