from src.data.data_loader import load_stock_data

from src.preprocessing.preprocess import (
    preprocess_data,
    save_processed_data
)

from src.features.feature_engineering import (
    create_features,
    save_feature_data
)


def main():

    print("Starting StockGo...")

    ticker = "AAPL"
    period = "1y"
    interval = "1d"

    # ==========================================================
    # STEP 1 — DOWNLOAD RAW DATA
    # ==========================================================

    data = load_stock_data(
        ticker=ticker,
        period=period,
        interval=interval
    )

    print("\nStock data downloaded successfully!")


    # ==========================================================
    # STEP 2 — PREPROCESS DATA
    # ==========================================================

    data = preprocess_data(data)

    print("\nData preprocessing completed!")


    save_processed_data(
        data=data,
        ticker=ticker,
        period=period
    )


    # ==========================================================
    # STEP 3 — FEATURE ENGINEERING
    # ==========================================================

    data = create_features(data)

    print("\nFeature engineering completed!")


    save_feature_data(
        data=data,
        ticker=ticker,
        period=period
    )


    # ==========================================================
    # STEP 4 — DISPLAY RESULTS
    # ==========================================================

    print("\nFinal dataset:")
    print(data.head())

    print("\nFinal dataset shape:")
    print(data.shape)

    print("\nFinal columns:")
    print(data.columns.tolist())


if __name__ == "__main__":
    main()