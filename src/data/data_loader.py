import yfinance as yf
import pandas as pd
from pathlib import Path


def load_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical stock market data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. AAPL, MSFT, TSLA.

    period : str
        Amount of historical data to download.

    interval : str
        Frequency of downloaded data.

    Returns
    -------
    pd.DataFrame
        Clean historical stock data.
    """

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data was downloaded for ticker: {ticker}"
        )

    # Handle yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    # Create raw data directory if it doesn't exist
    raw_directory = Path("data/raw")
    raw_directory.mkdir(parents=True, exist_ok=True)

    # Save downloaded data
    filename = f"{ticker}_{period}.csv"
    file_path = raw_directory / filename

    data.to_csv(file_path, index=False)

    print(f"Raw data saved to: {file_path}")

    return data