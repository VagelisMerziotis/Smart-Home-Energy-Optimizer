"""Streamlit AppTest coverage for navigation, headline metrics, and live controls.

These tests exercise the dashboard process, not a real browser. They
expect the saved Residual XGBoost headline figures (MAE 23.38 Wh, 2.75%
savings).
"""

import pytest

from streamlit.testing.v1 import AppTest

from config import BASE_DIR


APP_PATH = BASE_DIR / "app.py"


@pytest.fixture
def dashboard():

    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=60,
    )

    app.run()

    assert len(app.exception) == 0

    return app


def test_dashboard_initial_load(dashboard):

    assert len(dashboard.exception) == 0


@pytest.mark.parametrize(
    "page",
    [
        "Overview",
        "Live Simulation",
        "ML Evaluation",
        "Energy Evaluation",
        "System Architecture",
    ],
)
def test_dashboard_navigation(
    dashboard,
    page,
):

    dashboard.sidebar.radio[0].set_value(
        page
    ).run()

    assert len(dashboard.exception) == 0

    assert dashboard.sidebar.radio[0].value == page


def test_overview_metrics(dashboard):

    metrics = {
        metric.label: metric.value
        for metric in dashboard.metric
    }

    assert metrics["Test MAE"] == "23.38 Wh"
    assert metrics["Test RMSE"] == "59.34 Wh"
    assert metrics["Savings Rate"] == "2.75%"


def test_dashboard_single_step(dashboard):

    dashboard.sidebar.radio[0].set_value(
        "Live Simulation"
    ).run()

    step_button = next(
        button
        for button in dashboard.button
        if button.label == "⏭ Step"
    )

    step_button.click().run()

    assert len(dashboard.exception) == 0

    simulator = dashboard.session_state["simulator"]

    assert simulator.current_step == 1


def test_dashboard_reset(dashboard):

    dashboard.sidebar.radio[0].set_value(
        "Live Simulation"
    ).run()

    step_button = next(
        button
        for button in dashboard.button
        if button.label == "⏭ Step"
    )

    step_button.click().run()

    reset_button = next(
        button
        for button in dashboard.button
        if button.label == "↺ Reset"
    )

    reset_button.click().run()

    assert len(dashboard.exception) == 0

    simulator = dashboard.session_state["simulator"]

    assert simulator.current_step == 0
    assert simulator.get_history().empty


def test_dashboard_model_switch(dashboard):

    dashboard.sidebar.radio[0].set_value(
        "Live Simulation"
    ).run()

    model_selector = next(
        widget
        for widget in dashboard.selectbox
        if widget.label == "Simulation model"
    )

    model_selector.set_value(
        "Optimized XGBoost"
    ).run()

    assert len(dashboard.exception) == 0

    simulator = dashboard.session_state["simulator"]

    assert simulator.model_name == "Optimized XGBoost"
    assert simulator.current_step == 0