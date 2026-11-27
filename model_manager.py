"""Load a saved XGBoost artifact and produce non-negative Wh forecasts.

Direct models emit appliance consumption. Residual models emit the change
from `appliances_lag_0`; this manager reconstructs the absolute forecast as
max(0, lag_0 + residual). The default model name comes from
`results/model_selection.json`.
"""

import json

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from config import (
    MODEL_PATH,
    METADATA_PATH,
    OPTIMIZED_MODEL_PATH,
    OPTIMIZED_METADATA_PATH,
    RESIDUAL_MODEL_PATH,
    RESIDUAL_METADATA_PATH,
    MODEL_SELECTION_PATH,
    FORECAST_HORIZON_MINUTES,
)

from preprocessing import FEATURE_COLUMNS


# Maps dashboard/simulator labels to on-disk artifacts and how to decode
# the raw XGBoost output.
MODEL_REGISTRY = {
    "Original XGBoost": {
        "model_path": MODEL_PATH,
        "metadata_path": METADATA_PATH,
        "prediction_type": "direct",
    },
    "Optimized XGBoost": {
        "model_path": OPTIMIZED_MODEL_PATH,
        "metadata_path": OPTIMIZED_METADATA_PATH,
        "prediction_type": "direct",
    },
    "Residual XGBoost": {
        "model_path": RESIDUAL_MODEL_PATH,
        "metadata_path": RESIDUAL_METADATA_PATH,
        "prediction_type": "residual",
    },
}


class ModelManager:
    """Runtime wrapper around one registered booster plus its metadata."""

    def __init__(self, model_name=None):

        with open(
            MODEL_SELECTION_PATH,
            encoding="utf-8",
        ) as file:

            self.selection = json.load(file)

        self.model_name = (
            model_name
            or self.selection["selected_model"]
        )

        if self.model_name not in MODEL_REGISTRY:

            raise ValueError(
                f"Unknown model: {self.model_name}"
            )

        self.model_config = MODEL_REGISTRY[
            self.model_name
        ]

        self.prediction_type = self.model_config[
            "prediction_type"
        ]

        self.metadata = self.load_metadata()

        self.validate_metadata()

        self.model = self.load_model()

    def load_metadata(self):
        """Read the JSON sidecar stored next to the booster file."""


        path = self.model_config["metadata_path"]

        if not path.exists():

            raise FileNotFoundError(
                f"Metadata not found: {path}"
            )

        with open(
            path,
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def validate_metadata(self):
        """Reject models trained on a different feature set or horizon."""


        if (
            self.metadata["feature_columns"]
            != FEATURE_COLUMNS
        ):

            raise ValueError(
                "Model feature mismatch."
            )

        if (
            self.metadata["forecast_horizon_minutes"]
            != FORECAST_HORIZON_MINUTES
        ):

            raise ValueError(
                "Forecast horizon mismatch."
            )

        metadata_type = self.metadata.get(
            "prediction_type",
            "direct",
        )

        if metadata_type != self.prediction_type:

            raise ValueError(
                "Prediction type mismatch."
            )

    def load_model(self):
        """Load the XGBoost JSON booster from disk."""


        path = self.model_config["model_path"]

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        model = XGBRegressor()

        model.load_model(str(path))

        return model

    def prepare_features(self, features):
        """Select FEATURE_COLUMNS in training order and reject non-finite values."""


        if not isinstance(features, pd.DataFrame):

            raise TypeError(
                "Features must be a pandas DataFrame."
            )

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in features.columns
        ]

        if missing:

            raise ValueError(
                f"Missing features: {missing}"
            )

        features = features[
            FEATURE_COLUMNS
        ].copy()

        if not np.isfinite(
            features.to_numpy(dtype=float)
        ).all():

            raise ValueError(
                "Invalid feature values."
            )

        return features

    def predict(self, features):
        """Return clipped Wh forecasts for every row in `features`."""


        features = self.prepare_features(
            features
        )

        raw_predictions = np.asarray(
            self.model.predict(features),
            dtype=float,
        )

        if self.prediction_type == "residual":
            # Residual booster predicts y(t+1) - y(t); add current load back.
            baseline = features[
                "appliances_lag_0"
            ].to_numpy(dtype=float)

            predictions = (
                baseline + raw_predictions
            )

        else:

            predictions = raw_predictions

        if not np.isfinite(predictions).all():

            raise ValueError(
                "Non-finite model predictions."
            )

        return np.maximum(
            predictions,
            0.0,
        )

    def predict_one(self, features):
        """Return a single float forecast; `features` must contain one row."""


        if len(features) != 1:

            raise ValueError(
                "predict_one requires one row."
            )

        return float(
            self.predict(features)[0]
        )

    def get_model_info(self):
        """Return a JSON-serializable summary for logs and the dashboard."""


        return {
            "model_name": self.model_name,
            "prediction_type": self.prediction_type,
            "model_path": str(
                self.model_config["model_path"]
            ),
            "feature_count": len(FEATURE_COLUMNS),
            "forecast_horizon_minutes": (
                FORECAST_HORIZON_MINUTES
            ),
            "eco_threshold_wh": self.metadata[
                "eco_threshold_wh"
            ],
            "cv_mae_wh": self.metadata.get(
                "cv_mae_wh"
            ),
            "cv_rmse_wh": self.metadata.get(
                "cv_rmse_wh"
            ),
        }


def main():
    """Load the selected model and print the first test-set prediction."""


    from preprocessing import prepare_data

    manager = ModelManager()

    print("=" * 65)
    print("MODEL MANAGER")
    print("=" * 65)

    print(
        json.dumps(
            manager.get_model_info(),
            indent=4,
        )
    )

    _, test, _ = prepare_data()

    features = test.iloc[[0]][
        FEATURE_COLUMNS
    ]

    prediction = manager.predict_one(
        features
    )

    baseline = float(
        features["appliances_lag_0"].iloc[0]
    )

    print("\nFirst test prediction:")

    print(
        f"Persistence: {baseline:.4f} Wh"
    )

    print(
        f"Selected model: {prediction:.4f} Wh"
    )

    print(
        f"Actual consumption: "
        f"{test.iloc[0]['target']:.4f} Wh"
    )

    print("\nModel Manager: READY")


if __name__ == "__main__":
    main()
