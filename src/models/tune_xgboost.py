import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def tune_xgboost(
    data_path: str
):
    """
    Tune XGBoost hyperparameters using
    time-series cross-validation.
    """

    # ==========================================================
    # 1. LOAD DATA
    # ==========================================================

    data = pd.read_csv(data_path)

    print("Feature data loaded successfully!")
    print(f"Dataset shape: {data.shape}")


    # ==========================================================
    # 2. FEATURES
    # ==========================================================

    feature_columns = [
        "Open",
        "High",
        "Low",
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

    y = data["Target"]


    # ==========================================================
    # 3. TIME SERIES CROSS VALIDATION
    # ==========================================================

    tscv = TimeSeriesSplit(
        n_splits=4
    )


    # ==========================================================
    # 4. PARAMETER COMBINATIONS
    # ==========================================================

    parameter_sets = [
        {
            "n_estimators": 200,
            "max_depth": 2,
            "learning_rate": 0.03
        },
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03
        },
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03
        },
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05
        },
        {
            "n_estimators": 400,
            "max_depth": 2,
            "learning_rate": 0.05
        }
    ]


    # ==========================================================
    # 5. TEST PARAMETERS
    # ==========================================================

    results = []


    for params in parameter_sets:

        fold_mae = []
        fold_rmse = []
        fold_r2 = []


        print("\n----------------------------------------")
        print("Testing parameters:")
        print(params)
        print("----------------------------------------")


        for train_index, validation_index in tscv.split(X):

            X_train = X.iloc[train_index]
            X_validation = X.iloc[validation_index]

            y_train = y.iloc[train_index]
            y_validation = y.iloc[validation_index]


            model = XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )


            model.fit(
                X_train,
                y_train
            )


            predictions = model.predict(
                X_validation
            )


            mae = mean_absolute_error(
                y_validation,
                predictions
            )

            rmse = mean_squared_error(
                y_validation,
                predictions
            ) ** 0.5

            r2 = r2_score(
                y_validation,
                predictions
            )


            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)


        avg_mae = sum(fold_mae) / len(fold_mae)

        avg_rmse = sum(fold_rmse) / len(fold_rmse)

        avg_r2 = sum(fold_r2) / len(fold_r2)


        results.append({
            **params,
            "MAE": avg_mae,
            "RMSE": avg_rmse,
            "R2": avg_r2
        })


        print(
            f"Average MAE: {avg_mae:.6f}"
        )

        print(
            f"Average RMSE: {avg_rmse:.6f}"
        )

        print(
            f"Average R²: {avg_r2:.4f}"
        )


    # ==========================================================
    # 6. RESULTS
    # ==========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="MAE"
    )


    print("\n========================================")
    print("XGBoost Hyperparameter Results")
    print("========================================")

    print(
        results_df.to_string(
            index=False
        )
    )


    # ==========================================================
    # 7. BEST PARAMETERS
    # ==========================================================

    best = results_df.iloc[0]

    print("\n========================================")
    print("BEST PARAMETERS")
    print("========================================")

    print(
        f"n_estimators : {int(best['n_estimators'])}"
    )

    print(
        f"max_depth    : {int(best['max_depth'])}"
    )

    print(
        f"learning_rate: {best['learning_rate']}"
    )

    print(
        f"MAE          : {best['MAE']:.6f}"
    )

    print(
        f"RMSE         : {best['RMSE']:.6f}"
    )

    print(
        f"R²           : {best['R2']:.4f}"
    )


    return results_df