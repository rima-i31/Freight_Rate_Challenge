import pandas as pd


def load_data(path):
    """Load a CSV dataset and parse the date column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fit_preprocessing(df):
    """Fit preprocessing statistics on training data."""
    return {
        "weight_median": df["weight"].abs().median(),
        "market_index_median": df["market_index"].median(),
    }


def apply_preprocessing(df, stats):
    """Apply fitted preprocessing statistics to a dataset."""
    df = df.copy()

    if "weight" in df.columns:
        df["weight"] = df["weight"].abs()
        df["weight"] = df["weight"].fillna(
            stats["weight_median"]
        )

    if "market_index" in df.columns:
        df["market_index"] = df["market_index"].fillna(
            stats["market_index_median"]
        )

    return df