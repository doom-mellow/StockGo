import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


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


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

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
        ==
        (predictions > 0)
    ).mean()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy":
            directional_accuracy
    }


def compare_models(
    data_path,
    train_ratio=0.8
):

    data = pd.read_csv(
        data_path
    )

    data = data.sort_values(
        ["Date", "Ticker"]
    )

    X = data[
        FEATURE_COLUMNS
    ]

    y = data[
        "Target"
    ]

    split_index = int(
        len(data) * train_ratio
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

    results = []

    # ======================================================
    # NAIVE BASELINE
    # ======================================================

    baseline_predictions = pd.Series(
        0.0,
        index=y_test.index
    )

    results.append({
        "Model": "Naive Baseline",
        "MAE": mean_absolute_error(
            y_test,
            baseline_predictions
        ),
        "RMSE": mean_squared_error(
            y_test,
            baseline_predictions
        ) ** 0.5,
        "R2": r2_score(
            y_test,
            baseline_predictions
        ),
        "Directional Accuracy":
            (
                (y_test > 0)
                ==
                (baseline_predictions > 0)
            ).mean()
    })

    # ======================================================
    # RANDOM FOREST
    # ======================================================

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_results = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    rf_results["Model"] = (
        "Random Forest"
    )

    results.append(
        rf_results
    )

    # ======================================================
    # XGBOOST
    # ======================================================

    xgboost = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    xgboost.fit(
        X_train,
        y_train
    )

    xgb_results = evaluate_model(
        xgboost,
        X_test,
        y_test
    )

    xgb_results["Model"] = "XGBoost"

    results.append(
        xgb_results
    )

    # ======================================================
    # TUNED XGBOOST
    # ======================================================

    tuned_xgboost = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        objective="reg:squarederror",
        random_state=42
    )

    tuned_xgboost.fit(
        X_train,
        y_train
    )

    tuned_results = evaluate_model(
        tuned_xgboost,
        X_test,
        y_test
    )

    tuned_results["Model"] = (
        "Tuned XGBoost"
    )

    results.append(
        tuned_results
    )

    results_df = pd.DataFrame(
        results
    )

    return results_df