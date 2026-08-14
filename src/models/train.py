import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


def train_model(
    data_path: str,
    model_path: str = "models/saved_models/random_forest.pkl"
):
    """
    Train a Random Forest model to predict
    next-day stock returns.
    """

    # ==========================================================
    # 1. LOAD FEATURE DATA
    # ==========================================================

    data = pd.read_csv(data_path)

    print("Feature data loaded successfully!")
    print(f"Dataset shape: {data.shape}")


    # ==========================================================
    # 2. DEFINE FEATURES AND TARGET
    # ==========================================================

    feature_columns = [
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

    X = data[feature_columns]

    # Target = tomorrow's percentage return
    y = data["Target"]


    # ==========================================================
    # 3. TIME-BASED TRAIN / TEST SPLIT
    # ==========================================================

    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # ==========================================================
    # 4. CREATE MODEL
    # ==========================================================

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )


    # ==========================================================
    # 5. TRAIN MODEL
    # ==========================================================

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed!")


    # ==========================================================
    # 6. MAKE RETURN PREDICTIONS
    # ==========================================================

    predicted_returns = model.predict(X_test)


    # ==========================================================
    # 7. EVALUATE RETURN PREDICTIONS
    # ==========================================================

    mae = mean_absolute_error(
        y_test,
        predicted_returns
    )

    rmse = mean_squared_error(
        y_test,
        predicted_returns
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predicted_returns
    )


    print("\nReturn Prediction Evaluation")
    print("----------------------------")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.4f}")


    # ==========================================================
    # 8. CONVERT RETURNS TO PREDICTED PRICES
    # ==========================================================

    current_prices = data["Close"].iloc[split_index:]

    actual_prices = (
        data["Close"]
        .shift(-1)
        .iloc[split_index:]
    )

    # Remove the final row because there is
    # no next-day price for the final observation.
    valid_mask = actual_prices.notna()

    current_prices = current_prices[valid_mask]

    actual_prices = actual_prices[valid_mask]

    predicted_returns = predicted_returns[valid_mask]


    predicted_prices = (
        current_prices.values
        * (1 + predicted_returns)
    )


    # ==========================================================
    # 9. EVALUATE PRICE PREDICTIONS
    # ==========================================================

    price_mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    price_rmse = mean_squared_error(
        actual_prices,
        predicted_prices
    ) ** 0.5


    print("\nPrice Prediction Evaluation")
    print("---------------------------")
    print(f"MAE  : ${price_mae:.4f}")
    print(f"RMSE : ${price_rmse:.4f}")


    # ==========================================================
    # 10. SAVE MODEL
    # ==========================================================

    model_directory = Path(model_path).parent

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to: {model_path}"
    )


    # ==========================================================
    # 11. RETURN RESULTS
    # ==========================================================

    return (
        model,
        predicted_returns,
        y_test,
        predicted_prices,
        actual_prices
    )