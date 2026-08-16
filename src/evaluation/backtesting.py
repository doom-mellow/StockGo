import pandas as pd
import numpy as np


def backtest_strategy(
    data: pd.DataFrame,
    predictions,
    threshold: float = 0.0,
    transaction_cost: float = 0.001,
    initial_capital: float = 10000.0
):
    """
    Backtest a simple long-only StockGo trading strategy.

    Strategy
    --------
    - Go LONG when predicted return > threshold.
    - Stay in CASH otherwise.
    - Transaction costs are applied whenever the position changes.

    Parameters
    ----------
    data : pd.DataFrame
        Test dataset containing:
        - Date
        - Close
        - Target

    predictions : array-like
        Model-predicted next-day returns.

    threshold : float
        Minimum predicted return required to enter a trade.

    transaction_cost : float
        Cost applied when entering or exiting a position.
        Example:
            0.001 = 0.1%

    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict
        Dictionary containing performance metrics and equity curves.
    """

    data = data.copy().reset_index(drop=True)

    predictions = np.asarray(predictions, dtype=float)

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------

    if len(predictions) != len(data):
        raise ValueError(
            "Number of predictions must match number of rows in data."
        )

    required_columns = [
        "Date",
        "Close",
        "Target"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ----------------------------------------------------------
    # ACTUAL RETURNS
    # ----------------------------------------------------------

    actual_returns = (
        pd.to_numeric(
            data["Target"],
            errors="coerce"
        )
        .fillna(0)
        .values
    )

    # ----------------------------------------------------------
    # TRADING SIGNALS
    # ----------------------------------------------------------

    signals = np.where(
        predictions > threshold,
        1,
        0
    )

    # ----------------------------------------------------------
    # POSITION CHANGES
    # ----------------------------------------------------------

    previous_signal = np.concatenate(
        ([0], signals[:-1])
    )

    position_changes = (
        np.abs(signals - previous_signal)
    )

    # ----------------------------------------------------------
    # TRANSACTION COSTS
    # ----------------------------------------------------------

    costs = (
        position_changes
        * transaction_cost
    )

    # ----------------------------------------------------------
    # STRATEGY RETURNS
    # ----------------------------------------------------------

    strategy_returns_before_cost = (
        signals * actual_returns
    )

    strategy_returns = (
        strategy_returns_before_cost
        - costs
    )

    # ----------------------------------------------------------
    # BUY & HOLD RETURNS
    # ----------------------------------------------------------

    buy_hold_returns = actual_returns

    # ----------------------------------------------------------
    # EQUITY CURVES
    # ----------------------------------------------------------

    buy_hold_equity = (
        initial_capital
        * np.cumprod(
            1 + buy_hold_returns
        )
    )

    strategy_equity = (
        initial_capital
        * np.cumprod(
            1 + strategy_returns
        )
    )

    # ----------------------------------------------------------
    # FINAL VALUES
    # ----------------------------------------------------------

    final_buy_hold_value = (
        buy_hold_equity[-1]
    )

    final_strategy_value = (
        strategy_equity[-1]
    )

    # ----------------------------------------------------------
    # TOTAL RETURNS
    # ----------------------------------------------------------

    buy_hold_return = (
        final_buy_hold_value
        / initial_capital
        - 1
    )

    strategy_return = (
        final_strategy_value
        / initial_capital
        - 1
    )

    # ----------------------------------------------------------
    # NUMBER OF TRADES
    # ----------------------------------------------------------

    trades = int(
        position_changes.sum()
    )

    entries = int(
        ((signals == 1) &
         (previous_signal == 0)).sum()
    )

    exits = int(
        ((signals == 0) &
         (previous_signal == 1)).sum()
    )

    # ----------------------------------------------------------
    # WINNING / LOSING DAYS
    # ----------------------------------------------------------

    active_returns = strategy_returns[
        signals == 1
    ]

    winning_days = int(
        (active_returns > 0).sum()
    )

    losing_days = int(
        (active_returns < 0).sum()
    )

    if len(active_returns) > 0:

        win_rate = (
            winning_days
            / len(active_returns)
        )

    else:

        win_rate = 0.0

    # ----------------------------------------------------------
    # MARKET EXPOSURE
    # ----------------------------------------------------------

    exposure = (
        signals.mean()
    )

    # ----------------------------------------------------------
    # MAXIMUM DRAWDOWN
    # ----------------------------------------------------------

    running_max = np.maximum.accumulate(
        strategy_equity
    )

    drawdown = (
        strategy_equity
        - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    # ----------------------------------------------------------
    # VOLATILITY
    # ----------------------------------------------------------

    daily_volatility = np.std(
        strategy_returns,
        ddof=1
    )

    annualized_volatility = (
        daily_volatility
        * np.sqrt(252)
    )

    # ----------------------------------------------------------
    # SHARPE RATIO
    # ----------------------------------------------------------

    if daily_volatility != 0:

        sharpe_ratio = (
            np.mean(strategy_returns)
            / daily_volatility
        ) * np.sqrt(252)

    else:

        sharpe_ratio = 0.0

    # ----------------------------------------------------------
    # ANNUALIZED RETURN
    # ----------------------------------------------------------

    number_of_days = len(
        strategy_returns
    )

    if number_of_days > 0:

        annualized_return = (
            (final_strategy_value / initial_capital)
            ** (252 / number_of_days)
        ) - 1

    else:

        annualized_return = 0.0

    # ----------------------------------------------------------
    # CALMAR RATIO
    # ----------------------------------------------------------

    if max_drawdown != 0:

        calmar_ratio = (
            annualized_return
            / abs(max_drawdown)
        )

    else:

        calmar_ratio = 0.0

    # ----------------------------------------------------------
    # STRATEGY ADVANTAGE
    # ----------------------------------------------------------

    excess_return = (
        strategy_return
        - buy_hold_return
    )

    # ----------------------------------------------------------
    # EQUITY CURVE DATAFRAME
    # ----------------------------------------------------------

    equity_curve = pd.DataFrame({

        "Date": data["Date"],

        "Close": data["Close"],

        "Prediction": predictions,

        "Signal": signals,

        "Actual_Return": actual_returns,

        "Strategy_Return": strategy_returns,

        "Buy_Hold_Equity": buy_hold_equity,

        "StockGo_Equity": strategy_equity

    })

    # ----------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------

    results = {

        "Initial Capital":
            initial_capital,

        "Final Buy & Hold Value":
            final_buy_hold_value,

        "Final StockGo Value":
            final_strategy_value,

        "Buy & Hold Return":
            buy_hold_return,

        "StockGo Return":
            strategy_return,

        "Excess Return":
            excess_return,

        "Annualized Return":
            annualized_return,

        "Annualized Volatility":
            annualized_volatility,

        "Number of Trades":
            trades,

        "Entries":
            entries,

        "Exits":
            exits,

        "Winning Days":
            winning_days,

        "Losing Days":
            losing_days,

        "Win Rate":
            win_rate,

        "Market Exposure":
            exposure,

        "Maximum Drawdown":
            max_drawdown,

        "Sharpe Ratio":
            sharpe_ratio,

        "Calmar Ratio":
            calmar_ratio,

        "Transaction Cost":
            transaction_cost,

        "Equity Curve":
            equity_curve

    }

    return results


def print_backtest_results(results):
    """
    Print StockGo backtesting results.
    """

    print("\n")
    print("=" * 60)
    print("STOCKGO BACKTESTING RESULTS")
    print("=" * 60)

    print(
        f"Initial Capital       : "
        f"${results['Initial Capital']:,.2f}"
    )

    print(
        f"Final Buy & Hold      : "
        f"${results['Final Buy & Hold Value']:,.2f}"
    )

    print(
        f"Final StockGo Value   : "
        f"${results['Final StockGo Value']:,.2f}"
    )

    print("-" * 60)

    print(
        f"Buy & Hold Return     : "
        f"{results['Buy & Hold Return'] * 100:.2f}%"
    )

    print(
        f"StockGo Return        : "
        f"{results['StockGo Return'] * 100:.2f}%"
    )

    print(
        f"Excess Return         : "
        f"{results['Excess Return'] * 100:.2f}%"
    )

    print(
        f"Annualized Return     : "
        f"{results['Annualized Return'] * 100:.2f}%"
    )

    print(
        f"Annualized Volatility : "
        f"{results['Annualized Volatility'] * 100:.2f}%"
    )

    print("-" * 60)

    print(
        f"Number of Trades      : "
        f"{results['Number of Trades']}"
    )

    print(
        f"Entries               : "
        f"{results['Entries']}"
    )

    print(
        f"Exits                 : "
        f"{results['Exits']}"
    )

    print(
        f"Winning Days          : "
        f"{results['Winning Days']}"
    )

    print(
        f"Losing Days           : "
        f"{results['Losing Days']}"
    )

    print(
        f"Win Rate              : "
        f"{results['Win Rate'] * 100:.2f}%"
    )

    print(
        f"Market Exposure       : "
        f"{results['Market Exposure'] * 100:.2f}%"
    )

    print("-" * 60)

    print(
        f"Maximum Drawdown      : "
        f"{results['Maximum Drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe Ratio          : "
        f"{results['Sharpe Ratio']:.3f}"
    )

    print(
        f"Calmar Ratio          : "
        f"{results['Calmar Ratio']:.3f}"
    )

    print("-" * 60)

    print(
        f"Transaction Cost      : "
        f"{results['Transaction Cost'] * 100:.3f}%"
    )

    print("=" * 60)