import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_predictions(
    actual: pd.Series,
    predictions
):
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = mean_squared_error(
        actual,
        predictions
    ) ** 0.5

    r2 = r2_score(
        actual,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def return_baseline(
    data: pd.DataFrame,
    train_ratio: float = 0.8
):
    """
    Predict a 0% return for every test observation.

    This is equivalent to assuming that tomorrow's
    closing price will be today's closing price.
    """

    split_index = int(
        len(data) * train_ratio
    )

    actual_returns = data["Target"].iloc[split_index:]

    predicted_returns = pd.Series(
        0.0,
        index=actual_returns.index
    )

    metrics = evaluate_predictions(
        actual_returns,
        predicted_returns
    )

    return metrics


def get_feature_importance(
    model,
    feature_columns
):
    """
    Return model feature importance
    sorted from most important to least important.
    """

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    return importance


def directional_accuracy(
    actual_returns,
    predicted_returns
):
    """
    Calculate the percentage of predictions
    where the model correctly predicts the
    direction of the next-day return.
    """

    actual_direction = actual_returns > 0

    predicted_direction = predicted_returns > 0

    accuracy = (
        actual_direction == predicted_direction
    ).mean()

    return accuracy