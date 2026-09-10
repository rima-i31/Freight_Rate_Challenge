import pandas as pd


COMMON_FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "month",
    "day_of_week",
    "day_of_month",
    "days_since_start",
]


def create_features(df):
    """Create the feature set used by the final model."""
    features = df[COMMON_FEATURES].copy()

    features["date"] = pd.to_datetime(features["date"])

    features["month"] = features["date"].dt.month
    features["day_of_week"] = features["date"].dt.dayofweek
    features["day_of_month"] = features["date"].dt.day

    reference_date = pd.Timestamp("2025-01-01")
    features["days_since_start"] = (
        features["date"] - reference_date
    ).dt.days

    return features.drop(columns="date")