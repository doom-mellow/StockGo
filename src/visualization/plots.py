import pandas as pd


def get_feature_importance_data(model, feature_columns, top_n=10):
    """
    Extract and sort feature importance from a trained model.
    """

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    return importance.head(top_n)