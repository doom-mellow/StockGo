import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

from src.models.predict import (
    load_model,
    predict_next_day
)

from src.visualization.plots import (
    get_feature_importance_data
)

from src.evaluation.backtesting import (
    backtest_strategy,
    print_backtest_results
)

from src.evaluation.model_comparison import (
    compare_models
)

from src.evaluation.walk_forward import (
    walk_forward_validation
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="StockGo",
    page_icon="📈",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("📈 StockGo")

st.subheader(
    "AI-Powered Multi-Stock Return Prediction"
)

st.write(
    "StockGo uses machine learning and technical indicators "
    "to estimate the next trading day's stock return and price."
)


# ==========================================================
# FILE PATHS
# ==========================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_stock_2y_features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "xgboost_multi_stock.pkl"
)


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
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    return data


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_stock_model():

    return load_model(
        str(MODEL_PATH)
    )


# ==========================================================
# LOAD APPLICATION DATA
# ==========================================================

data = load_data()

model = load_stock_model()


# ==========================================================
# STOCK SELECTION
# ==========================================================

st.divider()

st.subheader("🎯 Select a Stock")

available_stocks = sorted(
    data["Ticker"].unique()
)

selected_stock = st.selectbox(
    "Choose a stock to analyze:",
    available_stocks
)


# ==========================================================
# FILTER STOCK
# ==========================================================

stock_data = data[
    data["Ticker"] == selected_stock
].copy()

stock_data = stock_data.sort_values(
    "Date"
).reset_index(drop=True)


# ==========================================================
# MAKE PREDICTION
# ==========================================================

result = predict_next_day(
    stock_data,
    model
)

current_price = result["current_price"]

predicted_return = result["predicted_return"]

predicted_price = result["predicted_price"]

direction = result["direction"]


# ==========================================================
# PREDICTION SECTION
# ==========================================================

st.divider()

st.header(
    f"📊 {selected_stock} Prediction"
)


# ==========================================================
# MAIN METRICS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Price",
        f"${current_price:.2f}"
    )


with col2:

    price_change = (
        predicted_price - current_price
    )

    st.metric(
        "Predicted Price",
        f"${predicted_price:.2f}",
        f"{price_change:+.2f}"
    )


with col3:

    st.metric(
        "Expected Return",
        f"{predicted_return * 100:.3f}%"
    )


with col4:

    st.metric(
        "Direction",
        direction
    )


# ==========================================================
# PREDICTION MESSAGE
# ==========================================================

if direction == "UP":

    st.success(
        f"📈 StockGo predicts that {selected_stock} "
        f"could move UP by approximately "
        f"{predicted_return * 100:.3f}% "
        f"on the next trading day."
    )

else:

    st.error(
        f"📉 StockGo predicts that {selected_stock} "
        f"could move DOWN by approximately "
        f"{abs(predicted_return) * 100:.3f}% "
        f"on the next trading day."
    )


# ==========================================================
# PRICE HISTORY
# ==========================================================

st.divider()

st.header(
    f"📈 {selected_stock} — Recent Price History"
)

recent_data = stock_data[
    ["Date", "Close"]
].tail(30)

recent_data = recent_data.set_index(
    "Date"
)

st.line_chart(
    recent_data["Close"]
)


# ==========================================================
# TECHNICAL INDICATORS
# ==========================================================

st.divider()

st.header("📐 Technical Indicators")

latest = stock_data.iloc[-1]

indicator_col1, indicator_col2, indicator_col3, indicator_col4 = (
    st.columns(4)
)


with indicator_col1:

    st.metric(
        "RSI (14)",
        f"{latest['RSI_14']:.2f}"
    )


with indicator_col2:

    st.metric(
        "SMA (20)",
        f"${latest['SMA_20']:.2f}"
    )


with indicator_col3:

    st.metric(
        "SMA (50)",
        f"${latest['SMA_50']:.2f}"
    )


with indicator_col4:

    st.metric(
        "20D Volatility",
        f"{latest['Volatility_20'] * 100:.2f}%"
    )


# ==========================================================
# MARKET DATA
# ==========================================================

st.divider()

st.header("📋 Recent Market Data")

display_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

recent_market_data = stock_data[
    display_columns
].tail(10).copy()

st.dataframe(
    recent_market_data,
    width="stretch"
)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

st.header("🤖 Model Information")

st.write(
    "StockGo uses an XGBoost regression model trained "
    "on historical data from multiple stocks."
)

st.write(
    "The model predicts the next trading day's percentage "
    "return using price, volume, momentum, volatility, "
    "moving averages, Bollinger Bands, RSI and MACD."
)

st.info(
    "The prediction is a next-day return estimate. "
    "The predicted price is calculated from the current "
    "price and predicted return."
)


# ==========================================================
# MODEL EXPLAINABILITY
# ==========================================================

st.divider()

st.header("🔍 Model Explainability")

st.write(
    "The following features had the greatest overall "
    "importance in the XGBoost model."
)

importance_data = get_feature_importance_data(
    model,
    FEATURE_COLUMNS,
    top_n=10
)

st.bar_chart(
    importance_data.set_index(
        "Feature"
    )["Importance"],
    horizontal=True
)

st.dataframe(
    importance_data,
    width="stretch"
)

st.caption(
    "Feature importance describes the model's overall use "
    "of each feature. It does not imply causation."
)


# ==========================================================
# BACKTESTING
# ==========================================================

st.divider()

st.header("📈 Strategy Backtesting")

st.write(
    "Backtesting evaluates how a simple StockGo trading "
    "strategy would have performed historically compared "
    "with a Buy & Hold strategy."
)


@st.cache_data
def run_backtest(selected_ticker):

    ticker_data = data[
        data["Ticker"] == selected_ticker
    ].copy()

    ticker_data = ticker_data.sort_values(
        "Date"
    ).reset_index(drop=True)

    split_index = int(
        len(ticker_data) * 0.8
    )

    test_data = ticker_data.iloc[
        split_index:
    ].copy()

    test_features = test_data[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        test_features
    )

    results = backtest_strategy(
        test_data,
        predictions
    )

    return results


backtest_results = run_backtest(
    selected_stock
)


# ==========================================================
# BACKTEST METRICS
# ==========================================================

initial_capital = backtest_results[
    "Initial Capital"
]

buy_hold_value = backtest_results[
    "Final Buy & Hold Value"
]

strategy_value = backtest_results[
    "Final StockGo Value"
]

buy_hold_return = backtest_results[
    "Buy & Hold Return"
]

strategy_return = backtest_results[
    "StockGo Return"
]

excess_return = backtest_results[
    "Excess Return"
]

annualized_return = backtest_results[
    "Annualized Return"
]

annualized_volatility = backtest_results[
    "Annualized Volatility"
]

trades = backtest_results[
    "Number of Trades"
]

entries = backtest_results[
    "Entries"
]

exits = backtest_results[
    "Exits"
]

winning_days = backtest_results[
    "Winning Days"
]

losing_days = backtest_results[
    "Losing Days"
]

win_rate = backtest_results[
    "Win Rate"
]

market_exposure = backtest_results[
    "Market Exposure"
]

max_drawdown = backtest_results[
    "Maximum Drawdown"
]

sharpe_ratio = backtest_results[
    "Sharpe Ratio"
]

calmar_ratio = backtest_results[
    "Calmar Ratio"
]

transaction_cost = backtest_results[
    "Transaction Cost"
]


# ==========================================================
# BACKTEST DISPLAY
# ==========================================================

st.subheader("💰 Portfolio Performance")

backtest_col1, backtest_col2, backtest_col3, backtest_col4 = (
    st.columns(4)
)

with backtest_col1:

    st.metric(
        "Initial Capital",
        f"${initial_capital:,.2f}"
    )

with backtest_col2:

    st.metric(
        "StockGo Final Value",
        f"${strategy_value:,.2f}"
    )

with backtest_col3:

    st.metric(
        "Buy & Hold Final Value",
        f"${buy_hold_value:,.2f}"
    )

with backtest_col4:

    st.metric(
        "StockGo Return",
        f"{strategy_return * 100:.2f}%"
    )


# ==========================================================
# RETURN METRICS
# ==========================================================

st.subheader("📊 Return & Risk Metrics")

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

with metric_col1:

    st.metric(
        "Buy & Hold Return",
        f"{buy_hold_return * 100:.2f}%"
    )

with metric_col2:

    st.metric(
        "Excess Return",
        f"{excess_return * 100:.2f}%"
    )

with metric_col3:

    st.metric(
        "Annualized Return",
        f"{annualized_return * 100:.2f}%"
    )

with metric_col4:

    st.metric(
        "Annualized Volatility",
        f"{annualized_volatility * 100:.2f}%"
    )


# ==========================================================
# STRATEGY METRICS
# ==========================================================

st.subheader("🎯 Trading Statistics")

strategy_col1, strategy_col2, strategy_col3, strategy_col4 = (
    st.columns(4)
)

with strategy_col1:

    st.metric(
        "Win Rate",
        f"{win_rate * 100:.2f}%"
    )

with strategy_col2:

    st.metric(
        "Market Exposure",
        f"{market_exposure * 100:.2f}%"
    )

with strategy_col3:

    st.metric(
        "Maximum Drawdown",
        f"{max_drawdown * 100:.2f}%"
    )

with strategy_col4:

    st.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.3f}"
    )


st.write(
    f"**Trades:** {trades}  |  "
    f"**Entries:** {entries}  |  "
    f"**Exits:** {exits}  |  "
    f"**Winning Days:** {winning_days}  |  "
    f"**Losing Days:** {losing_days}"
)

st.write(
    f"**Calmar Ratio:** {calmar_ratio:.3f}  |  "
    f"**Transaction Cost:** {transaction_cost * 100:.3f}%"
)


# ==========================================================
# EQUITY CURVE
# ==========================================================

st.subheader("📈 StockGo vs Buy & Hold")

equity_curve = backtest_results[
    "Equity Curve"
].copy()

equity_curve = equity_curve.set_index(
    "Date"
)

st.line_chart(
    equity_curve[
        [
            "Buy_Hold_Equity",
            "StockGo_Equity"
        ]
    ]
)


# ==========================================================
# BACKTEST INTERPRETATION
# ==========================================================

if strategy_return > buy_hold_return:

    st.success(
        "StockGo outperformed Buy & Hold during the "
        "selected historical test period."
    )

elif strategy_return < buy_hold_return:

    st.warning(
        "Buy & Hold outperformed StockGo during the "
        "selected historical test period."
    )

else:

    st.info(
        "StockGo and Buy & Hold produced similar returns "
        "during the selected test period."
    )


st.caption(
    "Historical backtest results do not guarantee future performance."
)
# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.divider()

st.header("⚖️ Model Comparison")

st.write(
    "StockGo compares multiple models against a naive "
    "zero-return baseline."
)


@st.cache_data
def load_model_comparison():

    return compare_models(
        str(DATA_PATH)
    )


try:

    comparison_results = load_model_comparison()

    st.dataframe(
        comparison_results,
        width="stretch"
    )

    st.subheader(
        "📊 Model MAE Comparison"
    )

    st.bar_chart(
        comparison_results.set_index(
            "Model"
        )["MAE"]
    )

    st.subheader(
        "🎯 Directional Accuracy"
    )

    st.bar_chart(
        comparison_results.set_index(
            "Model"
        )["Directional Accuracy"]
    )

except Exception as error:

    st.warning(
        f"Model comparison could not be loaded: {error}"
    )


# ==========================================================
# WALK-FORWARD VALIDATION
# ==========================================================

st.divider()

st.header("🔬 Walk-Forward Validation")

st.write(
    "Walk-forward validation evaluates the model using "
    "multiple chronological train/test windows. This gives "
    "a more realistic estimate of performance on unseen "
    "future observations."
)


@st.cache_data
def load_walk_forward():

    return walk_forward_validation(
        str(DATA_PATH)
    )


try:

    walk_forward_results = load_walk_forward()

    st.dataframe(
        walk_forward_results,
        width="stretch"
    )

    average_mae = (
        walk_forward_results["MAE"].mean()
    )

    average_rmse = (
        walk_forward_results["RMSE"].mean()
    )

    average_r2 = (
        walk_forward_results["R2"].mean()
    )

    average_directional = (
        walk_forward_results[
            "Directional Accuracy"
        ].mean()
    )

    wf_col1, wf_col2, wf_col3, wf_col4 = (
        st.columns(4)
    )

    with wf_col1:

        st.metric(
            "Average MAE",
            f"{average_mae:.6f}"
        )

    with wf_col2:

        st.metric(
            "Average RMSE",
            f"{average_rmse:.6f}"
        )

    with wf_col3:

        st.metric(
            "Average R²",
            f"{average_r2:.4f}"
        )

    with wf_col4:

        st.metric(
            "Directional Accuracy",
            f"{average_directional * 100:.2f}%"
        )

except Exception as error:

    st.warning(
        f"Walk-forward validation could not be loaded: {error}"
    )


# ==========================================================
# PROJECT SUMMARY
# ==========================================================

st.divider()

st.header("📚 What StockGo Does")

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.markdown(
        """
        **Data Pipeline**

        - Historical multi-stock market data
        - Data preprocessing
        - Technical indicator generation
        - Time-based train/test splitting
        - Multi-stock learning
        """
    )


with summary_col2:

    st.markdown(
        """
        **Machine Learning**

        - Random Forest baseline
        - XGBoost regression
        - Hyperparameter tuning
        - Feature importance
        - Walk-forward validation
        - Strategy backtesting
        """
    )


# ==========================================================
# LIMITATIONS
# ==========================================================

st.divider()

st.header("⚠️ Limitations")

st.write(
    "Stock prices are influenced by many factors that are "
    "not represented in this model, including news, earnings, "
    "macroeconomic conditions, market sentiment and unexpected "
    "events."
)

st.write(
    "The model therefore should not be interpreted as a "
    "guaranteed stock-picking system. Its predictions are "
    "estimates based primarily on historical price and volume "
    "patterns."
)


# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()

st.caption(
    "⚠️ StockGo is an educational machine-learning project. "
    "Predictions and backtests are estimates based on historical "
    "market data and should not be considered financial advice."
)