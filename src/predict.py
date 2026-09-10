from pathlib import Path
import joblib
import pandas as pd

from data_preparation import load_data, apply_preprocessing
from features import create_features


VALIDATION_PATH = Path("data/validation.csv")
VALIDATION_TEMPLATE_PATH = Path(
    "data/validation-predictions-template.csv"
)
DECEMBER_PATH = Path("data/december-chart-inputs.csv")

MODEL_PATH = Path("models/freight_rate_model.joblib")
STATS_PATH = Path("models/preprocessing_stats.joblib")

OUTPUT_PATH = Path("validation_predictions.csv")


def predict_dataset(df, pipeline, stats):
    clean_df = apply_preprocessing(df, stats)
    X = create_features(clean_df)

    return pipeline.predict(X)


def main():
    pipeline = joblib.load(MODEL_PATH)
    stats = joblib.load(STATS_PATH)

    # -------------------------
    # Final validation dataset
    # -------------------------

    validation = load_data(VALIDATION_PATH)

    validation_predictions = predict_dataset(
        validation,
        pipeline,
        stats,
    )

    template = pd.read_csv(VALIDATION_TEMPLATE_PATH)

    prediction_map = pd.Series(
        validation_predictions,
        index=validation["load_id"],
    )

    template["predicted_rate"] = (
        template["load_id"].map(prediction_map)
    )

    if template["predicted_rate"].isna().any():
        raise ValueError(
            "Some validation load_ids did not receive predictions."
        )

    template[["load_id", "predicted_rate"]].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------
    # Fixed December dataset
    # -------------------------

    december = load_data(DECEMBER_PATH)

    december["predicted_rate"] = predict_dataset(
        december,
        pipeline,
        stats,
    )

    december.to_csv(
        DECEMBER_PATH,
        index=False,
    )

    print(
        f"Saved {len(template):,} validation predictions "
        f"to {OUTPUT_PATH}"
    )

    print(
        f"Saved {len(december):,} December predictions "
        f"to {DECEMBER_PATH}"
    )


if __name__ == "__main__":
    main()