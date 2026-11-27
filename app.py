"""Streamlit dashboard for the Smart Home Energy Optimizer.

The app reads saved models and evaluation artifacts. Live Simulation
replays the test set in virtual time; it does not overwrite the official
results written by simulator.py. Start the app from the project root with
`python -m streamlit run app.py`.
"""

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    FINAL_FIGURES_DIR,
    FINAL_MODEL_COMPARISON_PATH,
    FINAL_CV_COMPARISON_PATH,
    FINAL_RANGE_METRICS_PATH,
    FINAL_FEATURE_IMPORTANCE_PATH,
    MODEL_SELECTION_PATH,
    ENERGY_POLICY_COMPARISON_PATH,
    SIMULATION_LOG_PATH,
    SIMULATION_METRICS_PATH,
    PERFORMANCE_METRICS_PATH,
    CONTROLLABLE_LOAD_RATIO,
    ECO_REDUCTION_RATIO,
    SIMULATION_STEP_SECONDS,
    FORECAST_HORIZON_MINUTES,
)

from simulator import EnergySimulator
from model_manager import MODEL_REGISTRY


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Home Energy Optimizer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #6EE7B7;
    }

    .system-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# CONSTANTS
# ==================================================

MODEL_COLORS = {
    "Persistence": "#94A3B8",
    "Original XGBoost": "#60A5FA",
    "Optimized XGBoost": "#FBBF24",
    "Residual XGBoost": "#34D399",
}

PAGES = [
    "Overview",
    "Live Simulation",
    "ML Evaluation",
    "Energy Evaluation",
    "System Architecture",
]


# ==================================================
# DATA LOADING
# ==================================================

@st.cache_data
def read_json_cached(path, modified_time):
    """Cache JSON by path and file mtime so edits on disk invalidate the cache."""

    with open(
        path,
        encoding="utf-8",
    ) as file:

        return json.load(file)


@st.cache_data
def read_csv_cached(
    path,
    modified_time,
    parse_dates=None,
):
    """Cache CSV tables the same way as JSON artifacts."""

    return pd.read_csv(
        path,
        parse_dates=parse_dates,
    )


def load_json(path):
    """Load JSON using the on-disk modification time as a cache key."""

    return read_json_cached(
        str(path),
        path.stat().st_mtime_ns,
    )


def load_csv(path, parse_dates=None):
    """Load a CSV using the on-disk modification time as a cache key."""

    return read_csv_cached(
        str(path),
        path.stat().st_mtime_ns,
        parse_dates,
    )


def load_artifacts():
    """Load the comparison tables and metrics shown on every dashboard page."""

    return {
        "selection": load_json(
            MODEL_SELECTION_PATH
        ),
        "test_metrics": load_csv(
            FINAL_MODEL_COMPARISON_PATH
        ),
        "cv_metrics": load_csv(
            FINAL_CV_COMPARISON_PATH
        ),
        "energy": load_csv(
            ENERGY_POLICY_COMPARISON_PATH
        ),
        "simulation_metrics": load_json(
            SIMULATION_METRICS_PATH
        ),
        "performance": load_json(
            PERFORMANCE_METRICS_PATH
        ),
        "range_metrics": load_csv(
            FINAL_RANGE_METRICS_PATH
        ),
        "feature_importance": load_csv(
            FINAL_FEATURE_IMPORTANCE_PATH
        ),
    }


try:

    artifacts = load_artifacts()

except (
    FileNotFoundError,
    ValueError,
    KeyError,
) as error:

    st.error(
        f"Required artifact unavailable: {error}"
    )

    st.info(
        "Run the training, simulation and final "
        "evaluation scripts before starting the app."
    )

    st.stop()


selected_model = artifacts[
    "selection"
]["selected_model"]

final_summary = artifacts[
    "simulation_metrics"
]

if final_summary["model_name"] != selected_model:

    st.error(
        "Saved simulation results do not match "
        "the selected final model."
    )

    st.stop()


# ==================================================
# FORMATTING HELPERS
# ==================================================

def format_wh(value):
    """Format a watt-hour value for Streamlit metrics."""

    return f"{value:,.2f} Wh"


def format_kwh(value):
    """Convert watt-hours to a kWh display string."""

    return f"{value / 1000:,.2f} kWh"


def format_percent(value):
    """Format a percentage already expressed on a 0–100 scale."""

    return f"{value:.2f}%"


def page_header(title, subtitle):
    """Render a page title and a one-line caption."""

    st.title(title)

    st.caption(subtitle)


def show_figure(filename):
    """Display a PNG from results/final_figures, or warn if it is missing."""

    path = FINAL_FIGURES_DIR / filename

    if path.exists():

        st.image(
            str(path),
            width="stretch",
        )

    else:

        st.warning(
            f"Figure not found: {filename}"
        )


def download_csv(
    dataframe,
    label,
    filename,
    key,
):
    """Offer a CSV download without rerunning the rest of the page."""

    st.download_button(
        label=label,
        data=dataframe.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        on_click="ignore",
    )


def simulation_disclaimer():
    """Remind readers that Eco savings are a modelling assumption."""

    st.info(
        "Energy savings are hypothetical and "
        "calculated using a fixed simulation policy. "
        "They are not measured savings from a "
        "physical smart home."
    )


def configure_chart(
    fig,
    title,
    y_title,
    height=420,
):
    """Apply the dark Plotly theme shared by live charts."""

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30,
        ),
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
    )

    return fig


# ==================================================
# SESSION STATE
# ==================================================
# Live replay lives in session_state so Start/Pause/Reset do not retrain
# or rewrite results/simulation_log.csv.

if "simulator" not in st.session_state:

    st.session_state.simulator = (
        EnergySimulator()
    )

if "running" not in st.session_state:

    st.session_state.running = False

if "speed" not in st.session_state:

    st.session_state.speed = (
        SIMULATION_STEP_SECONDS
    )

if "last_step_at" not in st.session_state:

    st.session_state.last_step_at = 0.0

if "live_model" not in st.session_state:

    st.session_state.live_model = (
        selected_model
    )


# ==================================================
# SIMULATION CALLBACKS
# ==================================================

def start_simulation():
    """Enable automatic virtual-time steps until pause, reset, or completion."""

    simulator = st.session_state.simulator

    if not simulator.finished:

        st.session_state.running = True

        st.session_state.last_step_at = (
            time.monotonic()
        )


def pause_simulation():
    """Stop automatic playback; history is kept."""

    st.session_state.running = False


def reset_simulation():
    """Clear the live session without rewriting official simulation files."""

    st.session_state.running = False

    st.session_state.simulator.reset()

    st.session_state.last_step_at = 0.0


def single_step():
    """Advance exactly one 10-minute interval while paused."""

    st.session_state.running = False

    simulator = st.session_state.simulator

    if not simulator.finished:

        simulator.step()

        st.session_state.last_step_at = (
            time.monotonic()
        )


def change_live_model():
    """Rebuild the live simulator when the user picks another booster."""

    st.session_state.running = False

    st.session_state.simulator = (
        EnergySimulator(
            model_name=(
                st.session_state.live_model
            )
        )
    )

    st.session_state.last_step_at = 0.0


def change_speed():
    """Reset the playback clock so the new interval starts from now."""

    st.session_state.last_step_at = (
        time.monotonic()
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🏠 Energy Optimizer")

    st.caption(
        "Intelligent Energy Management System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        PAGES,
    )

    st.divider()

    st.caption("FINAL SELECTED MODEL")

    st.success(selected_model)

    st.metric(
        "Eco Threshold",
        format_wh(
            final_summary["eco_threshold_wh"]
        ),
    )

    st.caption(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON_MINUTES} minutes"
    )

    st.caption(
        f"Controllable load: "
        f"{CONTROLLABLE_LOAD_RATIO:.0%}"
    )

    st.caption(
        f"Eco load reduction: "
        f"{ECO_REDUCTION_RATIO:.0%}"
    )

    st.divider()

    st.markdown(
        "**University of the Aegean**"
    )

    st.caption(
        "MSc — Pervasive Computing Systems"
    )

    st.caption(
        "Academic Year 2026–2027"
    )


# Pause live replay when navigating away.

if page != "Live Simulation":

    st.session_state.running = False


# ==================================================
# SHARED CHARTS
# ==================================================

def plot_model_metrics(data, key):
    """Grouped MAE/RMSE bars for a comparison table."""

    fig = go.Figure()

    for _, row in data.iterrows():

        fig.add_trace(
            go.Bar(
                name=row["Model"],
                x=["MAE", "RMSE"],
                y=[
                    row["MAE"],
                    row["RMSE"],
                ],
                marker_color=MODEL_COLORS[
                    row["Model"]
                ],
                text=[
                    f"{row['MAE']:.2f}",
                    f"{row['RMSE']:.2f}",
                ],
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group"
    )

    configure_chart(
        fig,
        "Regression Model Comparison",
        "Error (Wh)",
    )

    fig.update_yaxes(
        range=[
            0,
            data[["MAE", "RMSE"]]
            .to_numpy().max() * 1.25,
        ]
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
    )


def plot_live_consumption(
    history,
    threshold,
    model_name,
    window=144,
):
    """Plot the last 24 hours (144 steps) of the live replay."""

    data = history.tail(window)

    fig = go.Figure()

    traces = [
        (
            "actual_wh",
            "Actual consumption",
            "#F1F5F9",
            "solid",
        ),
        (
            "xgboost_prediction_wh",
            model_name,
            MODEL_COLORS[model_name],
            "solid",
        ),
        (
            "baseline_prediction_wh",
            "Persistence",
            "#94A3B8",
            "dash",
        ),
    ]

    for column, label, color, dash in traces:

        fig.add_trace(
            go.Scatter(
                x=data["target_time"],
                y=data[column],
                name=label,
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(
                    color=color,
                    width=2,
                    dash=dash,
                ),
            )
        )

    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text="Eco threshold",
    )

    configure_chart(
        fig,
        "Live Consumption Forecast",
        "Consumption (Wh)",
    )

    fig.update_layout(
        xaxis_title="Virtual target time"
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="live_forecast_chart",
    )


def plot_cumulative_energy(
    history,
    key,
):
    """Compare cumulative Always-Normal energy with the adaptive policy."""

    fig = go.Figure()

    normal = (
        history["always_normal_wh"]
        .cumsum() / 1000
    )

    adaptive = (
        history["adaptive_consumption_wh"]
        .cumsum() / 1000
    )

    fig.add_trace(
        go.Scatter(
            x=history["target_time"],
            y=normal,
            name="Always Normal",
            line=dict(
                color="#94A3B8",
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["target_time"],
            y=adaptive,
            name="Adaptive Normal/Eco",
            line=dict(
                color="#34D399",
                width=2,
            ),
            fill="tonexty",
            fillcolor="rgba(16,185,129,0.12)",
        )
    )

    configure_chart(
        fig,
        "Cumulative Energy Consumption",
        "Energy (kWh)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
    )


def plot_mode_distribution(
    summary,
    key,
):
    """Donut chart of Normal versus Eco interval counts."""

    fig = go.Figure(
        go.Pie(
            labels=["Normal", "Eco"],
            values=[
                summary["normal_intervals"],
                summary["eco_intervals"],
            ],
            hole=0.68,
            marker=dict(
                colors=[
                    "#64748B",
                    "#10B981",
                ]
            ),
            textinfo="label+percent",
        )
    )

    fig.update_layout(
        title="Operating Mode Distribution",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
        showlegend=False,
        annotations=[
            dict(
                text=(
                    f"{summary['intervals']:,}"
                    "<br>Intervals"
                ),
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )
        ],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
    )


# ==================================================
# PAGE 1 - OVERVIEW
# ==================================================

def overview_page():
    """Headline test metrics, hypothetical savings, and core latency."""

    page_header(
        "🏠 System Overview",
        "Machine learning and adaptive "
        "residential energy management",
    )

    test_metrics = artifacts[
        "test_metrics"
    ]

    final = test_metrics.loc[
        test_metrics["Model"]
        == selected_model
    ].iloc[0]

    baseline = test_metrics.loc[
        test_metrics["Model"]
        == "Persistence"
    ].iloc[0]

    mae_improvement = (
        (baseline["MAE"] - final["MAE"])
        / baseline["MAE"] * 100
    )

    rmse_improvement = (
        (baseline["RMSE"] - final["RMSE"])
        / baseline["RMSE"] * 100
    )

    st.subheader("Prediction Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Test MAE",
        format_wh(final["MAE"]),
        f"{mae_improvement:.2f}% lower vs baseline",
    )

    c2.metric(
        "Test RMSE",
        format_wh(final["RMSE"]),
        f"{rmse_improvement:.2f}% lower vs baseline",
    )

    c3.metric(
        "F1-score",
        f"{final['F1']:.2%}",
    )

    c4.metric(
        "CV MAE",
        format_wh(
            artifacts["selection"][
                "cv_mae_wh"
            ]
        ),
    )

    st.subheader("Energy Management")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recorded Consumption",
        format_kwh(
            final_summary[
                "always_normal_wh"
            ]
        ),
    )

    c2.metric(
        "Simulated Consumption",
        format_kwh(
            final_summary[
                "adaptive_consumption_wh"
            ]
        ),
    )

    c3.metric(
        "Hypothetical Savings",
        format_kwh(
            final_summary[
                "hypothetical_savings_wh"
            ]
        ),
    )

    c4.metric(
        "Savings Rate",
        format_percent(
            final_summary[
                "hypothetical_savings_percent"
            ]
        ),
    )

    left, right = st.columns([2, 1])

    with left:

        plot_model_metrics(
            test_metrics,
            key="overview_models",
        )

    with right:

        plot_mode_distribution(
            final_summary,
            key="overview_modes",
        )

    st.subheader("System Performance")

    performance = artifacts["performance"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Evaluated Intervals",
        f"{final_summary['intervals']:,}",
    )

    c2.metric(
        "Mean Core Latency",
        f"{performance['mean_ms']:.3f} ms",
    )

    c3.metric(
        "P99 Core Latency",
        f"{performance['p99_ms']:.3f} ms",
    )

    c4.metric(
        "Within 1 Second",
        format_percent(
            performance[
                "within_one_second_percent"
            ]
        ),
    )

    st.success(
        f"Final selected model: {selected_model}. "
        "Selection is based on time-series "
        "cross-validation MAE."
    )

    simulation_disclaimer()


# ==================================================
# PAGE 2 - LIVE SIMULATION
# ==================================================

def live_simulation_page():
    """Interactive virtual-time replay; session-only, not the official log."""

    page_header(
        "⚡ Live Simulation",
        "Virtual-time replay of historical "
        "energy consumption",
    )

    left, right = st.columns(2)

    with left:

        st.selectbox(
            "Simulation model",
            options=list(MODEL_REGISTRY),
            key="live_model",
            on_change=change_live_model,
            disabled=st.session_state.running,
        )

    with right:

        st.select_slider(
            "Seconds per virtual step",
            options=[
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
            ],
            key="speed",
            on_change=change_speed,
        )

    simulator = st.session_state.simulator

    if simulator.model_name != selected_model:

        st.warning(
            "Experimental model selected. "
            "The official final model and saved "
            "evaluation results remain unchanged."
        )

    c1, c2, c3, c4 = st.columns(4)

    c1.button(
        "▶ Start",
        on_click=start_simulation,
        disabled=(
            st.session_state.running
            or simulator.finished
        ),
        type="primary",
        width="stretch",
    )

    c2.button(
        "⏸ Pause",
        on_click=pause_simulation,
        disabled=(
            not st.session_state.running
        ),
        width="stretch",
    )

    c3.button(
        "⏭ Step",
        on_click=single_step,
        disabled=(
            st.session_state.running
            or simulator.finished
        ),
        width="stretch",
    )

    c4.button(
        "↺ Reset",
        on_click=reset_simulation,
        width="stretch",
    )

    st.caption(
        "Each virtual step represents 10 minutes "
        "of historical time. Playback speed is "
        "independent of the forecast horizon."
    )

    refresh_interval = (
        st.session_state.speed
        if st.session_state.running
        else None
    )

    @st.fragment(
        run_every=refresh_interval
    )
    def live_panel():

        simulator = (
            st.session_state.simulator
        )

        now = time.monotonic()

        elapsed = (
            now
            - st.session_state.last_step_at
        )

        # Advance only when the configured
        # playback interval has elapsed.

        if (
            st.session_state.running
            and not simulator.finished
            and elapsed >= (
                st.session_state.speed
            )
        ):

            simulator.step()

            st.session_state.last_step_at = (
                time.monotonic()
            )

        if (
            simulator.finished
            and st.session_state.running
        ):

            st.session_state.running = False

            st.rerun()

        if simulator.finished:

            st.success(
                "Simulation completed."
            )

        elif st.session_state.running:

            st.success(
                "Simulation running..."
            )

        else:

            st.info(
                "Simulation paused / ready."
            )

        st.progress(
            simulator.progress,
            text=(
                f"Step {simulator.current_step:,}"
                f" / {simulator.total_steps:,}"
            ),
        )

        history = simulator.get_history()

        if history.empty:

            st.info(
                "Press Start for automatic replay "
                "or Step to process one interval."
            )

            return

        latest = history.iloc[-1]

        st.subheader("Current Virtual Interval")

        st.caption(
            f"Decision time: "
            f"{latest['decision_time']}  |  "
            f"Target time: "
            f"{latest['target_time']}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Actual Consumption",
            format_wh(
                latest["actual_wh"]
            ),
        )

        c2.metric(
            "ML Prediction",
            format_wh(
                latest[
                    "xgboost_prediction_wh"
                ]
            ),
        )

        c3.metric(
            "Persistence",
            format_wh(
                latest[
                    "baseline_prediction_wh"
                ]
            ),
        )

        c4.metric(
            "Operating Mode",
            latest["mode"],
        )

        if latest["mode"] == "Eco":

            st.success(
                "ECO MODE — predicted consumption "
                "exceeds the configured threshold."
            )

        else:

            st.info(
                "NORMAL MODE — predicted consumption "
                "does not exceed the threshold."
            )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Prediction Error",
            format_wh(
                abs(
                    latest["actual_wh"]
                    - latest[
                        "xgboost_prediction_wh"
                    ]
                )
            ),
        )

        c2.metric(
            "Simulated Consumption",
            format_wh(
                latest[
                    "adaptive_consumption_wh"
                ]
            ),
        )

        c3.metric(
            "Interval Savings",
            format_wh(
                latest[
                    "hypothetical_savings_wh"
                ]
            ),
        )

        c4.metric(
            "Core Processing Time",
            f"{latest['processing_time_ms']:.3f} ms",
        )

        plot_live_consumption(
            history,
            simulator.controller.eco_threshold,
            simulator.model_name,
        )

        summary = simulator.get_summary()

        st.subheader("Session Energy Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Always Normal",
            format_kwh(
                summary["always_normal_wh"]
            ),
        )

        c2.metric(
            "Adaptive Consumption",
            format_kwh(
                summary[
                    "adaptive_consumption_wh"
                ]
            ),
        )

        c3.metric(
            "Hypothetical Savings",
            format_kwh(
                summary[
                    "hypothetical_savings_wh"
                ]
            ),
        )

        c4.metric(
            "Savings Rate",
            format_percent(
                summary[
                    "hypothetical_savings_percent"
                ]
            ),
        )

        left, right = st.columns([2, 1])

        with left:

            plot_cumulative_energy(
                history,
                key="live_cumulative",
            )

        with right:

            plot_mode_distribution(
                summary,
                key="live_modes",
            )

        with st.expander(
            "Recent Simulation Records",
            expanded=False,
        ):

            st.dataframe(
                history.tail(20),
                width="stretch",
                hide_index=True,
            )

        download_csv(
            history,
            "Download Session Log",
            "live_simulation_log.csv",
            "live_log_download",
        )

        simulation_disclaimer()

    live_panel()


# ==================================================
# PAGE 3 - ML EVALUATION
# ==================================================

def ml_evaluation_page():
    """CV selection, test errors, Eco detection, and feature importance."""

    page_header(
        "📊 Machine Learning Evaluation",
        "Chronological test evaluation and "
        "time-series cross-validation",
    )

    test_metrics = artifacts[
        "test_metrics"
    ]

    cv_metrics = artifacts[
        "cv_metrics"
    ]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Comparison",
        "Forecast Analysis",
        "Eco Detection",
        "Feature Importance",
        "Consumption Ranges",
    ])

    with tab1:

        st.subheader(
            "Final Model Selection"
        )

        st.success(
            f"{selected_model} was selected "
            "using the lowest mean CV MAE."
        )

        st.dataframe(
            cv_metrics.round(4),
            width="stretch",
            hide_index=True,
        )

        show_figure(
            "02_cv_model_comparison.png"
        )

        st.subheader(
            "Held-out Test Set Results"
        )

        st.dataframe(
            test_metrics[
                ["Model", "MAE", "RMSE"]
            ].round(4),
            width="stretch",
            hide_index=True,
        )

        plot_model_metrics(
            test_metrics,
            key="ml_test_metrics",
        )

        show_figure(
            "01_test_model_comparison.png"
        )

        download_csv(
            test_metrics,
            "Download Model Comparison",
            "model_comparison.csv",
            "ml_metrics_download",
        )

        st.caption(
            "The test period was previously examined "
            "during development. These results are "
            "not a new independent external validation."
        )

    with tab2:

        st.subheader(
            "Consumption Forecasts"
        )

        show_figure(
            "03_predictions_24h.png"
        )

        show_figure(
            "04_predictions_7days.png"
        )

        show_figure(
            "05_error_distribution.png"
        )

        show_figure(
            "06_actual_vs_predicted.png"
        )

    with tab3:

        st.subheader(
            "High Consumption Detection"
        )

        st.caption(
            "A positive event corresponds to "
            "consumption above the Eco threshold."
        )

        classification = test_metrics[
            [
                "Model",
                "Precision",
                "Recall",
                "F1",
                "TP",
                "TN",
                "FP",
                "FN",
            ]
        ].copy()

        for column in [
            "Precision",
            "Recall",
            "F1",
        ]:

            classification[column] = (
                classification[column]
                .map(
                    lambda value:
                    f"{value:.2%}"
                )
            )

        st.dataframe(
            classification,
            width="stretch",
            hide_index=True,
        )

        show_figure(
            "07_confusion_matrices.png"
        )

        show_figure(
            "08_classification_metrics.png"
        )

        st.info(
            "These classification metrics describe "
            "threshold detection accuracy. They do "
            "not measure real-world comfort or "
            "control effectiveness."
        )

    with tab4:

        st.subheader(
            "Normalized Gain Importance"
        )

        show_figure(
            "09_feature_importance.png"
        )

        st.dataframe(
            artifacts[
                "feature_importance"
            ].round(6),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "For Residual XGBoost, importance "
            "describes the learned residual model. "
            "Feature importance does not imply "
            "causation."
        )

    with tab5:

        st.subheader(
            "Performance by Consumption Level"
        )

        ranges = artifacts[
            "range_metrics"
        ]

        st.dataframe(
            ranges.round(4),
            width="stretch",
            hide_index=True,
        )

        for range_name in (
            ranges["Consumption_range"]
            .unique()
        ):

            subset = ranges.loc[
                ranges["Consumption_range"]
                == range_name
            ]

            st.markdown(
                f"**{range_name}**"
            )

            plot_model_metrics(
                subset,
                key=f"range_{range_name}",
            )

        download_csv(
            ranges,
            "Download Consumption Range Metrics",
            "consumption_range_metrics.csv",
            "range_download",
        )


# ==================================================
# PAGE 4 - ENERGY EVALUATION
# ==================================================

def energy_evaluation_page():
    """Official policy comparison and the saved simulation log."""

    page_header(
        "🌱 Energy Evaluation",
        "Comparison of simulated energy "
        "management policies",
    )

    summary = final_summary

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Always Normal",
        format_kwh(
            summary["always_normal_wh"]
        ),
    )

    c2.metric(
        "Adaptive Normal/Eco",
        format_kwh(
            summary[
                "adaptive_consumption_wh"
            ]
        ),
    )

    c3.metric(
        "Hypothetical Savings",
        format_kwh(
            summary[
                "hypothetical_savings_wh"
            ]
        ),
    )

    c4.metric(
        "Savings Rate",
        format_percent(
            summary[
                "hypothetical_savings_percent"
            ]
        ),
    )

    tab1, tab2, tab3 = st.tabs([
        "Policy Comparison",
        "Cumulative Energy",
        "Simulation Records",
    ])

    with tab1:

        energy = artifacts[
            "energy"
        ].copy()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=energy["Policy"],
                y=energy[
                    "Savings_percent"
                ],
                marker_color=[
                    "#10B981"
                    if policy == (
                        f"Adaptive {selected_model}"
                    )
                    else "#64748B"
                    for policy in energy["Policy"]
                ],
                text=[
                    f"{value:.2f}%"
                    for value in energy[
                        "Savings_percent"
                    ]
                ],
                textposition="outside",
            )
        )

        configure_chart(
            fig,
            "Hypothetical Savings by Policy",
            "Savings (%)",
            height=460,
        )

        fig.update_layout(
            hovermode="closest",
            xaxis_tickangle=-15,
        )

        fig.update_yaxes(
            range=[
                0,
                max(
                    energy[
                        "Savings_percent"
                    ].max() * 1.25,
                    1,
                ),
            ]
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="energy_policy_chart",
        )

        energy["Consumption_kWh"] = (
            energy["Consumption_Wh"]
            / 1000
        )

        energy["Savings_kWh"] = (
            energy["Savings_Wh"]
            / 1000
        )

        st.dataframe(
            energy[
                [
                    "Policy",
                    "Eco_intervals",
                    "Eco_percent",
                    "Consumption_kWh",
                    "Savings_kWh",
                    "Savings_percent",
                ]
            ].round(4),
            width="stretch",
            hide_index=True,
        )

        show_figure(
            "10_energy_policy_comparison.png"
        )

        download_csv(
            artifacts["energy"],
            "Download Energy Policy Comparison",
            "energy_policy_comparison.csv",
            "energy_policy_download",
        )

    simulation_log = load_csv(
        SIMULATION_LOG_PATH,
        parse_dates=[
            "decision_time",
            "target_time",
        ],
    )

    with tab2:

        plot_cumulative_energy(
            simulation_log,
            key="final_cumulative",
        )

        left, right = st.columns(2)

        with left:

            plot_mode_distribution(
                summary,
                key="final_mode_distribution",
            )

        with right:

            st.metric(
                "Normal Intervals",
                f"{summary['normal_intervals']:,}",
            )

            st.metric(
                "Eco Intervals",
                f"{summary['eco_intervals']:,}",
            )

            st.metric(
                "Eco Activation",
                format_percent(
                    summary["eco_percent"]
                ),
            )

        show_figure(
            "11_cumulative_energy.png"
        )

    with tab3:

        mode_filter = st.selectbox(
            "Filter by operating mode",
            ["All", "Normal", "Eco"],
        )

        filtered = simulation_log

        if mode_filter != "All":

            filtered = simulation_log.loc[
                simulation_log["mode"]
                == mode_filter
            ]

        st.caption(
            f"{len(filtered):,} records"
        )

        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
            height=500,
        )

        download_csv(
            filtered,
            "Download Filtered Records",
            "filtered_simulation_log.csv",
            "filtered_log_download",
        )

        download_csv(
            simulation_log,
            "Download Full Simulation Log",
            "simulation_log.csv",
            "full_log_download",
        )

    st.warning(
        "A policy that activates Eco more frequently "
        "can mechanically produce higher simulated "
        "savings. Higher savings alone do not establish "
        "a better prediction or control policy."
    )

    simulation_disclaimer()


# ==================================================
# PAGE 5 - SYSTEM ARCHITECTURE
# ==================================================

def architecture_page():
    """Pipeline diagram, residual equations, policy formula, and limitations."""

    page_header(
        "🧩 System Architecture",
        "Methodology, implementation and "
        "system limitations",
    )

    st.subheader(
        "Software Architecture"
    )

    st.graphviz_chart(
        """
        digraph Architecture {

            rankdir=TB;
            bgcolor="transparent";

            node [
                shape=box,
                style="rounded,filled",
                fillcolor="#1E293B",
                color="#64748B",
                fontcolor="white",
                fontname="Helvetica",
                margin=0.18
            ];

            edge [
                color="#94A3B8",
                arrowsize=0.7
            ];

            Dataset [
                label="UCI Energy Dataset"
            ];

            Loader [
                label="Data Loader & Validation"
            ];

            Features [
                label="Feature Engineering"
            ];

            Split [
                label="Chronological Train/Test Split"
            ];

            Training [
                label="XGBoost Training & Optimization"
            ];

            Selection [
                label="Model Selection (CV MAE)"
            ];

            Manager [
                label="Model Manager",
                fillcolor="#065F46"
            ];

            Forecast [
                label="10-Minute Forecast"
            ];

            Persistence [
                label="Persistence Baseline"
            ];

            Policy [
                label="Policy Engine"
            ];

            Normal [
                label="Normal Mode"
            ];

            Eco [
                label="Eco Mode",
                fillcolor="#065F46"
            ];

            Simulator [
                label="Virtual-Time Simulator"
            ];

            Evaluation [
                label="ML & Energy Evaluation"
            ];

            Logs [
                label="CSV / JSON Results"
            ];

            Dashboard [
                label="Streamlit Dashboard",
                fillcolor="#065F46"
            ];

            Dataset -> Loader;
            Loader -> Features;
            Features -> Split;
            Split -> Training;
            Training -> Selection;
            Selection -> Manager;
            Split -> Manager;
            Split -> Persistence;
            Manager -> Forecast;
            Forecast -> Policy;
            Policy -> Normal;
            Policy -> Eco;
            Normal -> Simulator;
            Eco -> Simulator;
            Split -> Simulator;
            Forecast -> Evaluation;
            Persistence -> Evaluation;
            Simulator -> Evaluation;
            Simulator -> Logs;
            Evaluation -> Logs;
            Logs -> Dashboard;
            Simulator -> Dashboard;
        }
        """,
        width="stretch",
    )

    st.subheader(
        "Prediction Methodology"
    )

    st.markdown(
        """
        The system predicts appliance energy
        consumption for the next 10-minute interval.

        The final model uses residual learning:
        it estimates the change in consumption
        relative to the most recent observation.
        """
    )

    st.latex(
        r"r_t = y_{t+1} - y_t"
    )

    st.latex(
        r"\hat{y}_{t+1} = "
        r"\max(0, y_t + \hat{r}_t)"
    )

    st.subheader(
        "Adaptive Energy Policy"
    )

    threshold = final_summary[
        "eco_threshold_wh"
    ]

    st.markdown(
        f"""
        - **Normal Mode:** forecast ≤ {threshold:g} Wh.
        - **Eco Mode:** forecast > {threshold:g} Wh.
        - Controllable load:
          {CONTROLLABLE_LOAD_RATIO:.0%}
          of recorded appliance consumption.
        - Eco reduction:
          {ECO_REDUCTION_RATIO:.0%}
          of the controllable load.
        """
    )

    st.latex(
        r"S_t = y_{t+1} \cdot "
        r"\alpha \cdot \beta \cdot "
        r"\mathbb{1}(\hat{y}_{t+1} > \theta)"
    )

    st.latex(
        r"E_{\mathrm{adaptive},t}"
        r" = y_{t+1} - S_t"
    )

    st.caption(
        "The actual future observation is revealed "
        "only after the operating mode is selected."
    )

    st.subheader(
        "Processing Performance"
    )

    performance = artifacts[
        "performance"
    ]

    latency_table = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "P95",
            "P99",
            "Maximum",
        ],
        "Latency (ms)": [
            performance["mean_ms"],
            performance["median_ms"],
            performance["p95_ms"],
            performance["p99_ms"],
            performance["max_ms"],
        ],
    })

    st.dataframe(
        latency_table.round(4),
        width="stretch",
        hide_index=True,
    )

    show_figure(
        "12_processing_performance.png"
    )

    st.caption(
        "Latency measurements cover the simulation "
        "core, not browser rendering, network delay "
        "or model training."
    )

    st.subheader(
        "Technologies"
    )

    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - XGBoost
        - Scikit-learn
        - Streamlit
        - Plotly and Matplotlib
        - Graphviz
        - CSV and JSON storage
        """
    )

    st.subheader(
        "Limitations"
    )

    st.markdown(
        """
        - Historical dataset from a single household.
        - No physical IoT sensors or actuators.
        - Appliance consumption does not necessarily
          represent total household consumption.
        - Fixed assumptions for controllable load
          and Eco reduction.
        - Hypothetical rather than measured savings.
        - No feedback from simulated consumption
          into future model inputs.
        - No direct measurement of user comfort.
        - The test set was previously examined
          during model development.
        - Model selection uses the same validation
          folds employed for hyperparameter tuning.
        """
    )

    st.subheader(
        "Dataset & References"
    )

    st.markdown(
        """
        **Dataset:** Appliances Energy Prediction,
        UCI Machine Learning Repository.

        **Reference:** Candanedo, L. M., Feldheim, V.,
        & Deramaix, D. (2017).
        Data driven prediction models of energy use
        of appliances in a low-energy house.
        *Energy and Buildings, 140*, 81–97.

        [UCI Dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

        [Research Paper](https://doi.org/10.1016/j.enbuild.2017.01.083)

        [XGBoost Documentation](https://xgboost.readthedocs.io/)

        [Streamlit Documentation](https://docs.streamlit.io/)
        """
    )


# ==================================================
# APPLICATION ROUTER
# ==================================================

PAGE_FUNCTIONS = {
    "Overview": overview_page,
    "Live Simulation": live_simulation_page,
    "ML Evaluation": ml_evaluation_page,
    "Energy Evaluation": energy_evaluation_page,
    "System Architecture": architecture_page,
}

PAGE_FUNCTIONS[page]()


st.divider()

st.caption(
    "Smart Home Energy Optimizer | "
    "University of the Aegean | "
    "MSc Project 2026–2027"
)
