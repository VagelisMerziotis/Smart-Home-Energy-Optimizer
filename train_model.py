"""Train the first (untuned) XGBoost model and compare it to persistence.

This is the baseline booster stored as models/xgboost_model.json. Later
scripts tune a direct model and a residual model; they do not replace this
file unless you re-run this script.
"""

import json
import time

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from config import (
    DATE_COLUMN,
    MODELS_DIR,
    RESULTS_DIR,
    MODEL_PATH,
    METADATA_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH,
    XGBOOST_PARAMS,
    FORECAST_HORIZON_MINUTES,
)

from preprocessing import (
    prepare_data,
    FEATURE_COLUMNS,
    TARGET_NAME,
    TARGET_TIME_COLUMN,
)


def train_xgboost(X_train, y_train):
    """Train the XGBoost regression model."""

    model = XGBRegressor(**XGBOOST_PARAMS)

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    return model, training_time


def persistence_predict(X):
    """Use Appliances at time t as the forecast for t + 10 minutes."""
    """Predict using the last known energy measurement."""

    return X["appliances_lag_0"].to_numpy()


def calculate_metrics(y_true, y_pred):
    """Calculate prediction error metrics in Wh."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def calculate_improvement(baseline_error, model_error):
    """Calculate percentage error reduction."""

    if baseline_error == 0:
        return None

    return float(
        (baseline_error - model_error)
        / baseline_error
        * 100
    )


def save_artifacts(
    model,
    train,
    test,
    eco_threshold,
    predictions,
    metrics,
):
    """Save model, metadata and evaluation results."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained XGBoost model
    model.save_model(str(MODEL_PATH))

    # Save model metadata
    metadata = {
        "model": "XGBRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "forecast_horizon_minutes": (
            FORECAST_HORIZON_MINUTES
        ),
        "eco_threshold_wh": float(eco_threshold),
        "training_samples": len(train),
        "test_samples": len(test),
        "train_start": str(
            train[TARGET_TIME_COLUMN].min()
        ),
        "train_end": str(
            train[TARGET_TIME_COLUMN].max()
        ),
        "test_start": str(
            test[TARGET_TIME_COLUMN].min()
        ),
        "test_end": str(
            test[TARGET_TIME_COLUMN].max()
        ),
        "model_parameters": XGBOOST_PARAMS,
    }

    with open(
        METADATA_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    # Save predictions
    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Save evaluation metrics
    with open(
        METRICS_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)


def print_results(metrics, eco_threshold):
    """Display model evaluation results."""

    print("\n" + "=" * 65)
    print("MODEL EVALUATION RESULTS")
    print("=" * 65)

    print(
        f"{'Model':<25}"
        f"{'MAE (Wh)':>15}"
        f"{'RMSE (Wh)':>15}"
    )

    print("-" * 65)

    for model_name in ["Persistence", "XGBoost"]:
        values = metrics[model_name]

        print(
            f"{model_name:<25}"
            f"{values['MAE']:>15.4f}"
            f"{values['RMSE']:>15.4f}"
        )

    print("-" * 65)

    print("\nError reduction compared to baseline:")

    for metric_name in ["MAE", "RMSE"]:
        improvement = metrics[
            "improvement_percent"
        ][metric_name]

        if improvement is None:
            print(f"{metric_name}: N/A")
        else:
            print(
                f"{metric_name}: "
                f"{improvement:.2f}%"
            )

    print(
        f"\nTraining time: "
        f"{metrics['training_time_seconds']:.4f} seconds"
    )

    print(
        f"Eco threshold: {eco_threshold:.2f} Wh"
    )

    print("\nSaved artifacts:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Metadata: {METADATA_PATH}")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    print(f"  Metrics: {METRICS_PATH}")

    print("\nTraining and evaluation: COMPLETED")
    print("=" * 65)


def main():
    print("=" * 65)
    print("SMART HOME ENERGY OPTIMIZER")
    print("XGBoost Training Pipeline")
    print("=" * 65)

    # Load prepared data
    train, test, eco_threshold = prepare_data()

    # Separate input features and target
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_NAME]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_NAME]

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    # Train XGBoost using training data only
    print("\nTraining XGBoost...")

    model, training_time = train_xgboost(
        X_train,
        y_train,
    )

    print("XGBoost training completed.")

    # Generate test predictions
    print("\nGenerating test predictions...")

    xgboost_predictions = model.predict(X_test)

    # Energy consumption cannot be negative
    xgboost_predictions = np.maximum(
        xgboost_predictions,
        0.0,
    )

    baseline_predictions = persistence_predict(
        X_test
    )

    # Evaluate both prediction methods
    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    xgboost_metrics = calculate_metrics(
        y_test,
        xgboost_predictions,
    )

    metrics = {
        "Persistence": baseline_metrics,
        "XGBoost": xgboost_metrics,
        "improvement_percent": {
            metric: calculate_improvement(
                baseline_metrics[metric],
                xgboost_metrics[metric],
            )
            for metric in ["MAE", "RMSE"]
        },
        "training_time_seconds": float(
            training_time
        ),
    }

    # Store predictions with their timestamps
    predictions = pd.DataFrame({
        "decision_time": test[DATE_COLUMN],
        "target_time": test[TARGET_TIME_COLUMN],
        "actual_wh": y_test.to_numpy(),
        "xgboost_prediction_wh": (
            xgboost_predictions
        ),
        "baseline_prediction_wh": (
            baseline_predictions
        ),
    })

    save_artifacts(
        model,
        train,
        test,
        eco_threshold,
        predictions,
        metrics,
    )

    print_results(metrics, eco_threshold)


if __name__ == "__main__":
    main()
