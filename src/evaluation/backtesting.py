import pandas as pd
import numpy as np


def backtest_strategy(
    data: pd.DataFrame,
    predictions,
    threshold: float = 0.0
):
    """
    Backtest a simple long-only stock prediction strategy.

    Strategy:
    - Buy/hold for the next trading day when predicted return
      is above the threshold.
    - Stay in cash when the predicted return is below the threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Test dataset containing Close prices and actual Target returns.

    predictions : array-like
        Model-predicted next-day returns.

    threshold : float
        Minimum predicted return required to enter a trade.

    Returns
    -------
    dict
        Backtesting performance metrics and equity curves.
    """

    data = data.copy().reset_index(drop=True)

    predictions = np.asarray(predictions)

    # Make sure prediction and data lengths match
    if len(predictions) != len(data):
        raise ValueError(
            "Number of predictions must match number of rows in data."
        )

    # ----------------------------------------------------------
    # 1. ACTUAL RETURNS
    # ----------------------------------------------------------

    actual_returns = data["Target"].astype(float).values

    # ----------------------------------------------------------
    # 2. GENERATE TRADING SIGNALS
    # ----------------------------------------------------------

    signals = np.where(
        predictions > threshold,
        1,
        0
    )

    # ----------------------------------------------------------
    # 3. STRATEGY RETURNS
    # ----------------------------------------------------------

    strategy_returns = signals * actual_returns

    # ----------------------------------------------------------
    # 4. EQUITY CURVES
    # ----------------------------------------------------------

    initial_capital = 10000

    buy_hold_equity = (
        initial_capital
        * np.cumprod(1 + actual_returns)
    )

    strategy_equity = (
        initial_capital
        * np.cumprod(1 + strategy_returns)
    )

    # ----------------------------------------------------------
    # 5. TOTAL RETURNS
    # ----------------------------------------------------------

    buy_hold_return = (
        buy_hold_equity[-1] / initial_capital - 1
    )

    strategy_return = (
        strategy_equity[-1] / initial_capital - 1
    )

    # ----------------------------------------------------------
    # 6. TRADING STATISTICS
    # ----------------------------------------------------------

    trades = signals.sum()

    winning_trades = (
        (strategy_returns > 0) & (signals == 1)
    ).sum()

    if trades > 0:
        win_rate = winning_trades / trades
    else:
        win_rate = 0.0

    # ----------------------------------------------------------
    # 7. MAXIMUM DRAWDOWN
    # ----------------------------------------------------------

    running_max = np.maximum.accumulate(
        strategy_equity
    )

    drawdown = (
        strategy_equity - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    # ----------------------------------------------------------
    # 8. SHARPE RATIO
    # ----------------------------------------------------------

    return_std = np.std(strategy_returns)

    if return_std != 0:
        sharpe_ratio = (
            np.mean(strategy_returns)
            / return_std
        ) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # ----------------------------------------------------------
    # 9. CREATE EQUITY CURVE DATAFRAME
    # ----------------------------------------------------------

    equity_curve = pd.DataFrame({
        "Date": data["Date"],
        "Buy_Hold": buy_hold_equity,
        "StockGo_Strategy": strategy_equity
    })

    # ----------------------------------------------------------
    # 10. RESULTS
    # ----------------------------------------------------------

    results = {
        "Initial Capital": initial_capital,
        "Final Buy & Hold Value": buy_hold_equity[-1],
        "Final StockGo Value": strategy_equity[-1],
        "Buy & Hold Return": buy_hold_return,
        "StockGo Return": strategy_return,
        "Number of Trades": int(trades),
        "Winning Trades": int(winning_trades),
        "Win Rate": win_rate,
        "Maximum Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Equity Curve": equity_curve
    }

    return results


def print_backtest_results(results):
    """
    Print backtesting results in a readable format.
    """

    print("\n")
    print("=" * 50)
    print("StockGo Backtesting Results")
    print("=" * 50)

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

    print(
        f"Buy & Hold Return     : "
        f"{results['Buy & Hold Return'] * 100:.2f}%"
    )

    print(
        f"StockGo Return        : "
        f"{results['StockGo Return'] * 100:.2f}%"
    )

    print(
        f"Number of Trades      : "
        f"{results['Number of Trades']}"
    )

    print(
        f"Winning Trades        : "
        f"{results['Winning Trades']}"
    )

    print(
        f"Win Rate              : "
        f"{results['Win Rate'] * 100:.2f}%"
    )

    print(
        f"Maximum Drawdown      : "
        f"{results['Maximum Drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe Ratio          : "
        f"{results['Sharpe Ratio']:.3f}"
    )

    print("=" * 50)