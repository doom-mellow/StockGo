import yfinance as yf
import pandas as pd
from pathlib import Path


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AMD"
]


def download_stock(
    ticker: str,
    period: str = "2y"
) -> pd.DataFrame:

    print(f"\nDownloading {ticker}...")

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for {ticker}"
        )

    # Handle yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Keep the columns we actually need
    required_columns = [
        "Date",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    data = data[required_columns]

    # Add ticker identifier
    data["Ticker"] = ticker

    return data


def download_multiple_stocks(
    tickers=None,
    period: str = "2y"
) -> pd.DataFrame:

    if tickers is None:
        tickers = DEFAULT_TICKERS

    all_data = []

    for ticker in tickers:

        try:

            data = download_stock(
                ticker,
                period
            )

            all_data.append(data)

            print(
                f"{ticker}: "
                f"{len(data)} rows downloaded"
            )

        except Exception as e:

            print(
                f"Failed to download "
                f"{ticker}: {e}"
            )

    if not all_data:
        raise ValueError(
            "No stock data was downloaded."
        )

    combined_data = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined_data


def save_multi_stock_data(
    data: pd.DataFrame,
    filename: str = "multi_stock_2y.csv"
):

    directory = Path("data/raw")

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = directory / filename

    data.to_csv(
        file_path,
        index=False
    )

    print(
        f"\nMulti-stock data saved to: "
        f"{file_path}"
    )

    return file_path