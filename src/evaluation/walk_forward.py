import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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


def directional_accuracy(actual, predicted):
    """
    Calculate percentage of predictions
    that correctly predict the direction.
    """

    actual_direction = np.asarray(actual) > 0
    predicted_direction = np.asarray(predicted) > 0

    return np.mean(
        actual_direction == predicted_direction
    )


def walk_forward_validation(
    data_path: str,
    n_splits: int = 5,
    test_size: int = 40
):
    """
    Perform walk-forward validation for the
    multi-stock XGBoost model.

    The model is repeatedly trained on past data
    and evaluated on future unseen data.
    """

    data = pd.read_csv(data_path)

    print("Feature data loaded successfully!")
    print(f"Dataset shape: {data.shape}")

    # ------------------------------------------------------
    # Sort by date
    # ------------------------------------------------------

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    results = []

    total_rows = len(data)

    # ------------------------------------------------------
    # Determine fold positions
    # ------------------------------------------------------

    initial_train_size = (
        total_rows - (n_splits * test_size)
    )

    if initial_train_size <= 0:
        raise ValueError(
            "Not enough data for the requested "
            "number of splits and test size."
        )

    # ------------------------------------------------------
    # WALK-FORWARD LOOP
    # ------------------------------------------------------

    for fold in range(n_splits):

        train_end = (
            initial_train_size
            + fold * test_size
        )

        test_start = train_end

        test_end = (
            test_start + test_size
        )

        train_data = data.iloc[:train_end]

        test_data = data.iloc[
            test_start:test_end
        ]

        if len(test_data) == 0:
            break

        X_train = train_data[
            FEATURE_COLUMNS
        ]

        y_train = train_data["Target"]

        X_test = test_data[
            FEATURE_COLUMNS
        ]

        y_test = test_data["Target"]

        # --------------------------------------------------
        # Train model
        # --------------------------------------------------

        model = XGBRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror"
        )

        model.fit(
            X_train,
            y_train
        )

        # --------------------------------------------------
        # Predict future data
        # --------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        direction = directional_accuracy(
            y_test,
            predictions
        )

        results.append({
            "Fold": fold + 1,
            "Train Samples": len(train_data),
            "Test Samples": len(test_data),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Directional Accuracy": direction
        })

        print()
        print(
            f"Fold {fold + 1}"
        )
        print("-" * 40)

        print(
            f"Training samples : "
            f"{len(train_data)}"
        )

        print(
            f"Testing samples  : "
            f"{len(test_data)}"
        )

        print(
            f"MAE              : "
            f"{mae:.6f}"
        )

        print(
            f"RMSE             : "
            f"{rmse:.6f}"
        )

        print(
            f"R²               : "
            f"{r2:.4f}"
        )

        print(
            f"Directional Acc. : "
            f"{direction * 100:.2f}%"
        )

    # ------------------------------------------------------
    # RESULTS DATAFRAME
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 60)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"Average MAE                 : "
        f"{results_df['MAE'].mean():.6f}"
    )

    print(
        f"Average RMSE                : "
        f"{results_df['RMSE'].mean():.6f}"
    )

    print(
        f"Average R²                  : "
        f"{results_df['R2'].mean():.4f}"
    )

    print(
        f"Average Directional Accuracy: "
        f"{results_df['Directional Accuracy'].mean() * 100:.2f}%"
    )

    print("=" * 60)

    return results_df