"""Full 3,947-step replay versus saved residual predictions and energy totals.

Always-Normal energy of 380,400 Wh and adaptive 369,922.2 Wh are the
published report figures for Residual XGBoost.
"""

import json

import numpy as np
import pandas as pd
import pytest

from config import (
    SIMULATION_METRICS_PATH,
    RESIDUAL_PREDICTIONS_PATH,
)

from simulator import EnergySimulator


@pytest.fixture(scope="module")
def completed_simulation():

    simulator = EnergySimulator()

    simulator.run_all()

    return simulator


def test_full_simulation_completion(
    completed_simulation,
):

    assert completed_simulation.finished

    assert completed_simulation.current_step == 3947

    assert len(
        completed_simulation.get_history()
    ) == 3947


def test_no_steps_after_completion(
    completed_simulation,
):

    assert completed_simulation.step() is None

    assert completed_simulation.current_step == 3947


def test_full_prediction_consistency(
    completed_simulation,
):

    history = completed_simulation.get_history()

    reference = pd.read_csv(
        RESIDUAL_PREDICTIONS_PATH
    )

    np.testing.assert_allclose(
        history["xgboost_prediction_wh"],
        reference["residual_prediction_wh"],
        rtol=1e-6,
        atol=1e-4,
    )


def test_final_energy_consumption(
    completed_simulation,
):

    summary = completed_simulation.get_summary()

    assert summary["always_normal_wh"] == pytest.approx(
        380400.0
    )

    assert summary[
        "adaptive_consumption_wh"
    ] == pytest.approx(369922.2)


def test_final_energy_savings(
    completed_simulation,
):

    summary = completed_simulation.get_summary()

    assert summary[
        "hypothetical_savings_wh"
    ] == pytest.approx(10477.8)

    assert summary[
        "hypothetical_savings_percent"
    ] == pytest.approx(
        10477.8 / 380400 * 100
    )


def test_final_mode_counts(
    completed_simulation,
):

    summary = completed_simulation.get_summary()

    assert summary["normal_intervals"] == 3020
    assert summary["eco_intervals"] == 927


def test_full_energy_balance(
    completed_simulation,
):

    history = completed_simulation.get_history()

    expected_savings = np.where(
        history["mode"] == "Eco",
        history["actual_wh"] * 0.30 * 0.20,
        0.0,
    )

    np.testing.assert_allclose(
        history["hypothetical_savings_wh"],
        expected_savings,
    )

    np.testing.assert_allclose(
        history["adaptive_consumption_wh"],
        history["actual_wh"] - expected_savings,
    )


def test_saved_metrics_consistency(
    completed_simulation,
):

    with open(
        SIMULATION_METRICS_PATH,
        encoding="utf-8",
    ) as file:

        saved = json.load(file)

    current = completed_simulation.get_summary()

    keys = [
        "intervals",
        "normal_intervals",
        "eco_intervals",
        "eco_percent",
        "always_normal_wh",
        "adaptive_consumption_wh",
        "hypothetical_savings_wh",
        "hypothetical_savings_percent",
    ]

    for key in keys:

        assert current[key] == pytest.approx(
            saved[key]
        )


def test_processing_time_requirement(
    completed_simulation,
):

    history = completed_simulation.get_history()

    assert (
        history["processing_time_ms"] < 1000
    ).all()