"""Build the final report figures from saved comparison and simulation files.

Does not retrain models. It checks that the simulator log, model selection
JSON, and ModelManager all refer to the same selected model.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

from config import (
    FINAL_FIGURES_DIR,
    FINAL_MODEL_COMPARISON_PATH,
    FINAL_CV_COMPARISON_PATH,
    RESIDUAL_PREDICTIONS_PATH,
    MODEL_SELECTION_PATH,
    ENERGY_POLICY_COMPARISON_PATH,
    SIMULATION_LOG_PATH,
    SIMULATION_METRICS_PATH,
    FINAL_RANGE_METRICS_PATH,
    FINAL_FEATURE_IMPORTANCE_PATH,
    PERFORMANCE_METRICS_PATH,
)

from model_manager import ModelManager


# --------------------------------------------------
# Visualization settings
# --------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

MODEL_COLORS = {
    "Persistence": "#64748b",
    "Original XGBoost": "#2563eb",
    "Optimized XGBoost": "#f59e0b",
    "Residual XGBoost": "#059669",
}

PREDICTION_COLUMNS = {
    "Persistence": "baseline_prediction_wh",
    "Original XGBoost": "original_prediction_wh",
    "Optimized XGBoost": "optimized_prediction_wh",
    "Residual XGBoost": "residual_prediction_wh",
}

SHORT_NAMES = {
    "Persistence": "Persistence",
    "Original XGBoost": "Original",
    "Optimized XGBoost": "Optimized",
    "Residual XGBoost": "Residual",
}


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_json(path):
    """Read a UTF-8 JSON artifact."""

    with open(
        path,
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_artifacts():
    """Load comparison tables, residual predictions, and simulation outputs."""

    return {
        "test_metrics": pd.read_csv(
            FINAL_MODEL_COMPARISON_PATH
        ),
        "cv_metrics": pd.read_csv(
            FINAL_CV_COMPARISON_PATH
        ),
        "predictions": pd.read_csv(
            RESIDUAL_PREDICTIONS_PATH,
            parse_dates=[
                "decision_time",
                "target_time",
            ],
        ),
        "energy": pd.read_csv(
            ENERGY_POLICY_COMPARISON_PATH
        ),
        "simulation": pd.read_csv(
            SIMULATION_LOG_PATH,
            parse_dates=[
                "decision_time",
                "target_time",
            ],
        ),
        "selection": load_json(
            MODEL_SELECTION_PATH
        ),
        "simulation_metrics": load_json(
            SIMULATION_METRICS_PATH
        ),
    }


def save_figure(fig, filename):
    """Write a PNG under results/final_figures and close the figure."""

    path = FINAL_FIGURES_DIR / filename

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {filename}")


# --------------------------------------------------
# 01 / 02 - Model comparisons
# --------------------------------------------------

def plot_model_comparison(
    data,
    title,
    filename,
):
    """Side-by-side MAE and RMSE bars for each candidate model."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    names = [
        SHORT_NAMES[name]
        for name in data["Model"]
    ]

    colors = [
        MODEL_COLORS[name]
        for name in data["Model"]
    ]

    for ax, metric in zip(
        axes,
        ["MAE", "RMSE"],
    ):

        bars = ax.bar(
            names,
            data[metric],
            color=colors,
            width=0.65,
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=4,
        )

        ax.set_title(metric)

        ax.set_ylabel("Error (Wh)")

        ax.set_ylim(
            0,
            data[metric].max() * 1.20,
        )

        ax.tick_params(
            axis="x",
            rotation=15,
        )

        ax.grid(axis="x", visible=False)

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )

    save_figure(fig, filename)


# --------------------------------------------------
# 03 / 04 - Time-series predictions
# --------------------------------------------------

def plot_predictions(
    data,
    selected_model,
    samples,
    title,
    filename,
):
    """Overlay actual, persistence, and selected-model forecasts."""

    subset = data.head(samples)

    fig, ax = plt.subplots(
        figsize=(15, 5),
    )

    ax.plot(
        subset["target_time"],
        subset["actual_wh"],
        color="#111827",
        linewidth=1.6,
        label="Actual consumption",
    )

    ax.plot(
        subset["target_time"],
        subset["baseline_prediction_wh"],
        color=MODEL_COLORS["Persistence"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="Persistence",
    )

    ax.plot(
        subset["target_time"],
        subset[
            PREDICTION_COLUMNS[selected_model]
        ],
        color=MODEL_COLORS[selected_model],
        linewidth=1.4,
        label=selected_model,
    )

    ax.set_title(title)

    ax.set_xlabel("Target timestamp")

    ax.set_ylabel("Consumption (Wh)")

    locator = mdates.AutoDateLocator()

    ax.xaxis.set_major_locator(locator)

    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    ax.legend(loc="upper right")

    save_figure(fig, filename)


# --------------------------------------------------
# 05 - Prediction error distribution
# --------------------------------------------------

def plot_error_distribution(data):
    """Overlay signed error histograms for all four prediction methods."""

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    errors = {
        name: (
            data["actual_wh"] - data[column]
        )
        for name, column in
        PREDICTION_COLUMNS.items()
    }

    combined = np.concatenate([
        values.to_numpy()
        for values in errors.values()
    ])

    bins = np.linspace(
        combined.min(),
        combined.max(),
        100,
    )

    for name, values in errors.items():

        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.7,
            color=MODEL_COLORS[name],
            label=name,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Prediction Error Distribution"
    )

    ax.set_xlabel(
        "Actual - Predicted (Wh)"
    )

    ax.set_ylabel("Frequency")

    ax.legend()

    save_figure(
        fig,
        "05_error_distribution.png",
    )


# --------------------------------------------------
# 06 - Actual vs predicted
# --------------------------------------------------

def plot_actual_vs_predicted(
    data,
    selected_model,
):
    """Scatter of predicted versus actual Wh for the selected model."""

    actual = data["actual_wh"]

    predicted = data[
        PREDICTION_COLUMNS[selected_model]
    ]

    fig, ax = plt.subplots(
        figsize=(7, 7),
    )

    ax.scatter(
        actual,
        predicted,
        alpha=0.25,
        s=14,
        color=MODEL_COLORS[selected_model],
        edgecolors="none",
    )

    maximum = max(
        actual.max(),
        predicted.max(),
    )

    ax.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--",
        color="#111827",
        label="Perfect prediction",
    )

    ax.set_xlim(0, maximum * 1.05)
    ax.set_ylim(0, maximum * 1.05)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        f"Actual vs Predicted\n"
        f"{selected_model}"
    )

    ax.set_xlabel("Actual consumption (Wh)")

    ax.set_ylabel(
        "Predicted consumption (Wh)"
    )

    ax.legend()

    save_figure(
        fig,
        "06_actual_vs_predicted.png",
    )


# --------------------------------------------------
# 07 - Confusion matrices
# --------------------------------------------------

def plot_confusion_matrices(metrics):
    """2x2 Eco/Normal detection matrices derived from thresholded forecasts."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 10),
    )

    vmax = metrics[
        ["TN", "FP", "FN", "TP"]
    ].to_numpy().max()

    for ax, (_, row) in zip(
        axes.flat,
        metrics.iterrows(),
    ):

        matrix = np.array([
            [row["TN"], row["FP"]],
            [row["FN"], row["TP"]],
        ], dtype=int)

        ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )

        for i in range(2):

            for j in range(2):

                value = matrix[i, j]

                ax.text(
                    j,
                    i,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=17,
                    fontweight="bold",
                    color=(
                        "white"
                        if value > vmax / 2
                        else "#111827"
                    ),
                )

        ax.set_xticks(
            [0, 1],
            ["Normal", "Eco"],
        )

        ax.set_yticks(
            [0, 1],
            ["Low", "High"],
        )

        ax.set_xlabel("Predicted mode")

        ax.set_ylabel("Actual consumption")

        ax.set_title(row["Model"])

        ax.grid(False)

    fig.suptitle(
        "High Consumption Detection",
        fontsize=16,
        fontweight="bold",
    )

    save_figure(
        fig,
        "07_confusion_matrices.png",
    )


# --------------------------------------------------
# 08 - Classification metrics
# --------------------------------------------------

def plot_classification_metrics(metrics):
    """Precision, recall, and F1 for detecting intervals above the Eco cutoff."""

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    x = np.arange(3)

    width = 0.19

    for i, (_, row) in enumerate(
        metrics.iterrows()
    ):

        values = [
            row["Precision"] * 100,
            row["Recall"] * 100,
            row["F1"] * 100,
        ]

        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            color=MODEL_COLORS[row["Model"]],
            label=row["Model"],
        )

        ax.bar_label(
            bars,
            fmt="%.1f",
            fontsize=8,
            padding=3,
        )

    ax.set_xticks(
        x,
        ["Precision", "Recall", "F1-score"],
    )

    ax.set_ylim(0, 115)

    ax.set_ylabel("Score (%)")

    ax.set_title(
        "High Consumption Detection Metrics"
    )

    ax.legend(
        loc="upper center",
        ncol=2,
    )

    ax.grid(axis="x", visible=False)

    save_figure(
        fig,
        "08_classification_metrics.png",
    )


# --------------------------------------------------
# 09 - Final model feature importance
# --------------------------------------------------

def plot_feature_importance(manager):
    """Gain importance of the selected booster; residual models rank residual features."""

    manager.model.importance_type = "gain"

    importance = pd.Series(
        manager.model.feature_importances_,
        index=manager.metadata[
            "feature_columns"
        ],
        name="importance",
    ).sort_values()

    importance.sort_values(
        ascending=False
    ).to_csv(
        FINAL_FEATURE_IMPORTANCE_PATH,
        index_label="feature",
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    bars = ax.barh(
        importance.index,
        importance.values,
        color=MODEL_COLORS[
            manager.model_name
        ],
    )

    ax.bar_label(
        bars,
        fmt="%.4f",
        padding=4,
        fontsize=9,
    )

    ax.set_xlim(
        0,
        max(
            importance.max() * 1.25,
            0.1,
        ),
    )

    ax.set_title(
        f"Feature Importance (Gain)\n"
        f"{manager.model_name}"
    )

    ax.set_xlabel(
        "Normalized gain importance"
    )

    ax.grid(axis="y", visible=False)

    save_figure(
        fig,
        "09_feature_importance.png",
    )

    print("\nFinal feature importance:")

    print(
        importance.sort_values(
            ascending=False
        ).round(4).to_string()
    )


# --------------------------------------------------
# 10 - Energy policy comparison
# --------------------------------------------------

def plot_energy_comparison(
    energy,
    selected_model,
):
    """Hypothetical kWh and percent savings for each policy, highlighting the winner."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    labels = energy["Policy"].str.replace(
        "Adaptive ",
        "",
        regex=False,
    )

    colors = [
        "#059669"
        if name == f"Adaptive {selected_model}"
        else "#94a3b8"
        for name in energy["Policy"]
    ]

    bars = axes[0].barh(
        labels,
        energy["Consumption_Wh"] / 1000,
        color=colors,
    )

    axes[0].bar_label(
        bars,
        fmt="%.2f",
        padding=4,
    )

    axes[0].set_xlim(
        0,
        energy["Consumption_Wh"].max()
        / 1000 * 1.18,
    )

    axes[0].set_title(
        "Simulated Energy Consumption"
    )

    axes[0].set_xlabel("Energy (kWh)")

    bars = axes[1].barh(
        labels,
        energy["Savings_percent"],
        color=colors,
    )

    axes[1].bar_label(
        bars,
        fmt="%.2f%%",
        padding=4,
    )

    axes[1].set_xlim(
        0,
        max(
            energy["Savings_percent"].max()
            * 1.25,
            1,
        ),
    )

    axes[1].set_title(
        "Hypothetical Energy Savings"
    )

    axes[1].set_xlabel("Savings (%)")

    for ax in axes:

        ax.invert_yaxis()

        ax.grid(axis="y", visible=False)

    fig.suptitle(
        "Energy Policy Comparison",
        fontsize=16,
        fontweight="bold",
    )

    save_figure(
        fig,
        "10_energy_policy_comparison.png",
    )


# --------------------------------------------------
# 11 - Cumulative energy
# --------------------------------------------------

def plot_cumulative_energy(
    simulation,
    selected_model,
):
    """Running Always-Normal versus adaptive energy over the test period."""

    fig, ax = plt.subplots(
        figsize=(13, 6),
    )

    normal = (
        simulation["always_normal_wh"]
        .cumsum() / 1000
    )

    adaptive = (
        simulation["adaptive_consumption_wh"]
        .cumsum() / 1000
    )

    ax.plot(
        simulation["target_time"],
        normal,
        color="#64748b",
        linewidth=2,
        label="Always Normal",
    )

    ax.plot(
        simulation["target_time"],
        adaptive,
        color="#059669",
        linewidth=2,
        label=f"Adaptive {selected_model}",
    )

    ax.fill_between(
        simulation["target_time"],
        adaptive,
        normal,
        color="#059669",
        alpha=0.15,
        label="Hypothetical savings",
    )

    ax.set_title(
        "Cumulative Simulated Energy Consumption"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Cumulative energy (kWh)"
    )

    ax.legend()

    fig.autofmt_xdate()

    save_figure(
        fig,
        "11_cumulative_energy.png",
    )


# --------------------------------------------------
# 12 - Processing performance
# --------------------------------------------------

def plot_processing_performance(simulation):
    """Core step latency distribution and write performance_metrics.json."""

    latency = simulation[
        "processing_time_ms"
    ].to_numpy()

    metrics = {
        "samples": len(latency),
        "mean_ms": float(np.mean(latency)),
        "median_ms": float(np.median(latency)),
        "p95_ms": float(
            np.percentile(latency, 95)
        ),
        "p99_ms": float(
            np.percentile(latency, 99)
        ),
        "max_ms": float(np.max(latency)),
        "min_ms": float(np.min(latency)),
        "within_one_second_percent": float(
            np.mean(latency <= 1000) * 100
        ),
    }

    with open(
        PERFORMANCE_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].hist(
        latency,
        bins=60,
        color="#2563eb",
        edgecolor="white",
    )

    axes[0].set_yscale("log")

    axes[0].set_title(
        "Processing Time Distribution"
    )

    axes[0].set_xlabel("Latency (ms)")

    axes[0].set_ylabel(
        "Frequency (log scale)"
    )

    labels = [
        "Median",
        "Mean",
        "P95",
        "P99",
        "Maximum",
    ]

    values = [
        metrics["median_ms"],
        metrics["mean_ms"],
        metrics["p95_ms"],
        metrics["p99_ms"],
        metrics["max_ms"],
    ]

    bars = axes[1].bar(
        labels,
        values,
        color="#059669",
    )

    axes[1].bar_label(
        bars,
        fmt="%.2f",
        padding=3,
    )

    axes[1].set_ylim(
        0,
        max(values) * 1.2,
    )

    axes[1].set_title(
        "Processing Time Statistics"
    )

    axes[1].set_ylabel("Latency (ms)")

    axes[1].grid(
        axis="x",
        visible=False,
    )

    save_figure(
        fig,
        "12_processing_performance.png",
    )

    print("\nProcessing performance:")

    print(
        json.dumps(
            metrics,
            indent=4,
        )
    )


# --------------------------------------------------
# Consumption range analysis
# --------------------------------------------------

def analyze_consumption_ranges(
    predictions,
    threshold,
):
    """MAE/RMSE split by whether actual consumption is above the Eco threshold."""

    ranges = {
        "Low consumption": (
            predictions["actual_wh"]
            <= threshold
        ),
        "High consumption": (
            predictions["actual_wh"]
            > threshold
        ),
    }

    rows = []

    for range_name, mask in ranges.items():

        subset = predictions.loc[mask]

        if subset.empty:

            continue

        actual = subset["actual_wh"]

        for name, column in (
            PREDICTION_COLUMNS.items()
        ):

            errors = actual - subset[column]

            rows.append({
                "Consumption_range": range_name,
                "Model": name,
                "Samples": len(subset),
                "MAE": float(
                    np.abs(errors).mean()
                ),
                "RMSE": float(
                    np.sqrt(
                        np.square(errors).mean()
                    )
                ),
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        FINAL_RANGE_METRICS_PATH,
        index=False,
    )

    print("\nConsumption range analysis:")

    print(
        result.round(4)
        .to_string(index=False)
    )


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():
    """Validate selected-model consistency and write the final PNG set."""

    FINAL_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SMART HOME ENERGY OPTIMIZER")
    print("Final Evaluation & Visualization")
    print("=" * 70)

    artifacts = load_artifacts()

    selected_model = artifacts[
        "selection"
    ]["selected_model"]

    manager = ModelManager()

    if manager.model_name != selected_model:

        raise ValueError(
            "Selected model mismatch."
        )

    if (
        artifacts["simulation_metrics"][
            "model_name"
        ] != selected_model
    ):

        raise ValueError(
            "Simulation model mismatch."
        )

    if not artifacts["simulation"][
        "model_name"
    ].eq(selected_model).all():

        raise ValueError(
            "Simulation log model mismatch."
        )

    print(
        f"Selected model: {selected_model}"
    )

    print(
        f"Test samples: "
        f"{len(artifacts['predictions']):,}"
    )

    plot_model_comparison(
        artifacts["test_metrics"],
        "Test Set Model Comparison",
        "01_test_model_comparison.png",
    )

    plot_model_comparison(
        artifacts["cv_metrics"],
        "Time Series Cross-Validation",
        "02_cv_model_comparison.png",
    )

    plot_predictions(
        artifacts["predictions"],
        selected_model,
        samples=144,
        title="Consumption Forecast - First 24 Hours",
        filename="03_predictions_24h.png",
    )

    plot_predictions(
        artifacts["predictions"],
        selected_model,
        samples=1008,
        title="Consumption Forecast - First 7 Days",
        filename="04_predictions_7days.png",
    )

    plot_error_distribution(
        artifacts["predictions"]
    )

    plot_actual_vs_predicted(
        artifacts["predictions"],
        selected_model,
    )

    plot_confusion_matrices(
        artifacts["test_metrics"]
    )

    plot_classification_metrics(
        artifacts["test_metrics"]
    )

    plot_feature_importance(manager)

    plot_energy_comparison(
        artifacts["energy"],
        selected_model,
    )

    plot_cumulative_energy(
        artifacts["simulation"],
        selected_model,
    )

    plot_processing_performance(
        artifacts["simulation"]
    )

    analyze_consumption_ranges(
        artifacts["predictions"],
        manager.metadata[
            "eco_threshold_wh"
        ],
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION: COMPLETED")
    print("=" * 70)

    print(
        f"Figures directory: "
        f"{FINAL_FIGURES_DIR}"
    )

    print(
        f"Generated figures: "
        f"{len(list(FINAL_FIGURES_DIR.glob('*.png')))}"
    )


if __name__ == "__main__":
    main()
