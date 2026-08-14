import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


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
# TRAIN MODEL
# ==========================================================

def train_xgboost(
    data_path: str,
    model_path: str = (
        "models/saved_models/"
        "xgboost_multi_stock.pkl"
    )
):

    # ======================================================
    # 1. LOAD DATA
    # ======================================================

    data = pd.read_csv(data_path)

    print(
        "Feature data loaded successfully!"
    )

    print(
        f"Dataset shape: {data.shape}"
    )

    # ======================================================
    # 2. SORT BY DATE
    # ======================================================

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    # ======================================================
    # 3. DEFINE FEATURES AND TARGET
    # ======================================================

    X = data[FEATURE_COLUMNS]

    y = data["Target"]

    # ======================================================
    # 4. TIME-BASED TRAIN / TEST SPLIT
    # ======================================================

    split_index = int(
        len(data) * 0.8
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # ======================================================
    # 5. CREATE XGBOOST MODEL
    # ======================================================

    model = XGBRegressor(

        n_estimators=200,

        max_depth=2,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=-1
    )

    # ======================================================
    # 6. TRAIN
    # ======================================================

    print(
        "\nTraining Multi-Stock XGBoost..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Multi-stock XGBoost "
        "training completed!"
    )

    # ======================================================
    # 7. PREDICTIONS
    # ======================================================

    predictions = model.predict(
        X_test
    )

    # ======================================================
    # 8. EVALUATION
    # ======================================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    directional_accuracy = (
        (y_test > 0)
        == (predictions > 0)
    ).mean()

    print(
        "\n## Multi-Stock XGBoost "
        "Evaluation"
    )

    print(
        "--------------------------------"
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"Directional Accuracy: "
        f"{directional_accuracy * 100:.2f}%"
    )

    # ======================================================
    # 9. SAVE MODEL
    # ======================================================

    model_directory = Path(
        model_path
    ).parent

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to: "
        f"{model_path}"
    )

    return (
        model,
        predictions,
        y_test
    )