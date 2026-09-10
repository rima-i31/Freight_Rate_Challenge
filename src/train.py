from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from data_preparation import (
    load_data,
    fit_preprocessing,
    apply_preprocessing,
)
from features import (
    create_features,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


DATA_PATH = Path("data/train-test.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "freight_rate_model.joblib"
STATS_PATH = MODEL_DIR / "preprocessing_stats.joblib"

TARGET = "posted_rate"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    df = load_data(DATA_PATH)

    preprocessing_stats = fit_preprocessing(df)
    clean_df = apply_preprocessing(
        df,
        preprocessing_stats,
    )

    X = create_features(clean_df)
    y = clean_df[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(preprocessing_stats, STATS_PATH)

    print(f"Model trained on {len(df):,} rows.")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()