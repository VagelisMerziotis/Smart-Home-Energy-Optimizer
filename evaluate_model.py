"""Plot the original (untuned) XGBoost against persistence on the test set.

Reads models/xgboost_model.json and results/predictions.csv. Final report
figures are produced by evaluate_final.py instead.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from config import (
    MODEL_PATH,
    METADATA_PATH,
    PREDICTIONS_PATH,
    METRICS_PATH,
    RESULTS_DIR,
)


FIGURES_DIR = RESULTS_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.2,
})


def load_results():
    """Load saved predictions, metrics and metadata."""

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=[
            "decision_time",
            "target_time",
        ],
    )

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    return predictions, metrics, metadata


def save_figure(fig, filename):
    """Save a figure and release memory."""

    path = FIGURES_DIR / filename

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_model_comparison(metrics):
    """Compare MAE and RMSE."""

    models = ["Persistence", "XGBoost"]

    mae = [
        metrics[model]["MAE"]
        for model in models
    ]

    rmse = [
        metrics[model]["RMSE"]
        for model in models
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(models))
    width = 0.32

    bars_mae = ax.bar(
        x - width / 2,
        mae,
        width,
        label="MAE",
        color="#5087bd",
    )

    bars_rmse = ax.bar(
        x + width / 2,
        rmse,
        width,
        label="RMSE",
        color="#64748b",
    )

    ax.bar_label(bars_mae, fmt="%.2f", padding=3)
    ax.bar_label(bars_rmse, fmt="%.2f", padding=3)

    ax.set_xticks(x, models)
    ax.set_ylabel("Prediction error (Wh)")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0, max(mae + rmse) * 1.2)
    ax.legend()

    save_figure(fig, "01_model_comparison.png")


def plot_predictions(predictions, samples=144):
    """Plot the first 24 hours of the test set."""

    data = predictions.head(samples)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        data["target_time"],
        data["actual_wh"],
        label="Actual consumption",
        color="#222222",
        linewidth=1.8,
    )

    ax.plot(
        data["target_time"],
        data["xgboost_prediction_wh"],
        label="XGBoost",
        color="#5087bd",
        linewidth=1.5,
    )

    ax.plot(
        data["target_time"],
        data["baseline_prediction_wh"],
        label="Persistence",
        color="#e69f00",
        linestyle="--",
        alpha=0.8,
    )

    ax.set_title(
        "Actual vs Predicted Consumption — First 24 Hours"
    )

    ax.set_xlabel("Target time")
    ax.set_ylabel("Energy consumption (Wh)")
    ax.legend()

    fig.autofmt_xdate()

    save_figure(fig, "02_predictions_24h.png")


def plot_error_distribution(predictions):
    """Compare signed prediction errors."""

    xgb_errors = (
        predictions["actual_wh"]
        - predictions["xgboost_prediction_wh"]
    )

    baseline_errors = (
        predictions["actual_wh"]
        - predictions["baseline_prediction_wh"]
    )

    bins = np.histogram_bin_edges(
        np.concatenate([
            xgb_errors.to_numpy(),
            baseline_errors.to_numpy(),
        ]),
        bins=60,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        baseline_errors,
        bins=bins,
        alpha=0.5,
        label="Persistence",
        color="#64748b",
    )

    ax.hist(
        xgb_errors,
        bins=bins,
        alpha=0.5,
        label="XGBoost",
        color="#5087bd",
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
    )

    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Actual − Predicted (Wh)")
    ax.set_ylabel("Frequency")
    ax.legend()

    save_figure(fig, "03_error_distribution.png")


def plot_actual_vs_predicted(predictions):
    """Visualize XGBoost prediction accuracy."""

    actual = predictions["actual_wh"]
    predicted = predictions["xgboost_prediction_wh"]

    max_value = max(
        actual.max(),
        predicted.max(),
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        actual,
        predicted,
        alpha=0.25,
        s=12,
        color="#5087bd",
    )

    ax.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
        color="black",
        label="Perfect prediction",
    )

    ax.set_xlim(0, max_value * 1.05)
    ax.set_ylim(0, max_value * 1.05)
    ax.set_aspect("equal")

    ax.set_xlabel("Actual consumption (Wh)")
    ax.set_ylabel("Predicted consumption (Wh)")
    ax.set_title("XGBoost: Actual vs Predicted")
    ax.legend()

    save_figure(fig, "04_actual_vs_predicted.png")


def plot_feature_importance(metadata):
    """Plot XGBoost gain-based feature importance."""

    model = XGBRegressor(
        importance_type="gain"
    )

    model.load_model(str(MODEL_PATH))

    importance = pd.Series(
        model.feature_importances_,
        index=metadata["feature_columns"],
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))

    importance.plot(
        kind="barh",
        ax=ax,
        color="#5087bd",
    )

    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Relative importance (gain)")
    ax.set_ylabel("Input feature")

    save_figure(fig, "05_feature_importance.png")

    importance.sort_values(
        ascending=False
    ).rename("importance").to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index_label="feature",
    )

    print("\nFeature importance ranking:")
    print(
        importance.sort_values(
            ascending=False
        ).round(4)
    )


def analyze_consumption_ranges(
    predictions,
    threshold,
):
    """Compare errors below and above the Eco threshold."""

    groups = {
        "Low consumption": predictions[
            predictions["actual_wh"] <= threshold
        ],
        "High consumption": predictions[
            predictions["actual_wh"] > threshold
        ],
    }

    rows = []

    for name, data in groups.items():

        if data.empty:
            continue

        xgb_error = (
            data["actual_wh"]
            - data["xgboost_prediction_wh"]
        )

        baseline_error = (
            data["actual_wh"]
            - data["baseline_prediction_wh"]
        )

        rows.append({
            "consumption_range": name,
            "samples": len(data),
            "persistence_mae_wh": (
                baseline_error.abs().mean()
            ),
            "xgboost_mae_wh": (
                xgb_error.abs().mean()
            ),
            "persistence_rmse_wh": np.sqrt(
                (baseline_error ** 2).mean()
            ),
            "xgboost_rmse_wh": np.sqrt(
                (xgb_error ** 2).mean()
            ),
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        RESULTS_DIR / "consumption_range_metrics.csv",
        index=False,
    )

    print("\nPerformance by consumption range:")
    print(
        summary.round(4).to_string(index=False)
    )


def main():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions, metrics, metadata = load_results()

    print("=" * 65)
    print("SMART HOME ENERGY OPTIMIZER")
    print("ML Evaluation and Visualization")
    print("=" * 65)

    print(f"Test samples: {len(predictions):,}\n")

    plot_model_comparison(metrics)

    plot_predictions(predictions)

    plot_error_distribution(predictions)

    plot_actual_vs_predicted(predictions)

    plot_feature_importance(metadata)

    analyze_consumption_ranges(
        predictions,
        metadata["eco_threshold_wh"],
    )

    print("\nEvaluation: COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
