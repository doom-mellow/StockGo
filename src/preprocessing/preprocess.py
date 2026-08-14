import pandas as pd
from pathlib import Path


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess stock market data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw stock market data.

    Returns
    -------
    pd.DataFrame
        Cleaned stock market data.
    """

    data = data.copy()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Remove column index name
    data.columns.name = None

    # Convert Date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort data chronologically
    data = data.sort_values("Date")

    # Reset index
    data = data.reset_index(drop=True)

    # Remove missing values
    data = data.dropna()

    return data


def save_processed_data(
    data: pd.DataFrame,
    ticker: str,
    period: str
):
    """
    Save cleaned stock data to the processed data directory.
    """

    processed_directory = Path("data/processed")
    processed_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = f"{ticker}_{period}_clean.csv"

    file_path = processed_directory / filename

    data.to_csv(
        file_path,
        index=False
    )

    print(
        f"Processed data saved to: {file_path}"
    )