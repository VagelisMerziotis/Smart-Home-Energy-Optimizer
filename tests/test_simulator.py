"""Stepwise simulator behaviour: chronology, Eco math, reset, and replay.

The first test interval is 2016-04-30 08:10 → 08:20, actual 370 Wh,
persistence 300 Wh. Residual XGBoost predicts above 100 Wh, so Eco
savings are 370 * 0.30 * 0.20 = 22.2 Wh.
"""

import numpy as np
import pandas as pd
import pytest

from simulator import EnergySimulator


@pytest.fixture
def simulator():
    return EnergySimulator()


def test_initial_state(simulator):

    assert simulator.current_step == 0
    assert simulator.total_steps == 3947
    assert not simulator.finished
    assert simulator.get_history().empty


def test_single_step(simulator):

    record = simulator.step()

    assert record is not None
    assert simulator.current_step == 1
    assert len(simulator.get_history()) == 1


def test_multiple_steps(simulator):

    for _ in range(10):
        simulator.step()

    assert simulator.current_step == 10
    assert len(simulator.get_history()) == 10


def test_reset(simulator):

    for _ in range(5):
        simulator.step()

    simulator.reset()

    assert simulator.current_step == 0
    assert simulator.get_history().empty
    assert not simulator.finished


def test_first_interval(simulator):

    record = simulator.step()

    assert pd.Timestamp(
        record["decision_time"]
    ) == pd.Timestamp("2016-04-30 08:10:00")

    assert pd.Timestamp(
        record["target_time"]
    ) == pd.Timestamp("2016-04-30 08:20:00")

    assert record["actual_wh"] == pytest.approx(370)

    assert record[
        "baseline_prediction_wh"
    ] == pytest.approx(300)


def test_first_interval_energy(simulator):

    record = simulator.step()

    assert record["mode"] == "Eco"

    assert record[
        "hypothetical_savings_wh"
    ] == pytest.approx(22.2)

    assert record[
        "adaptive_consumption_wh"
    ] == pytest.approx(347.8)


def test_chronological_replay(simulator):

    for _ in range(50):
        simulator.step()

    history = simulator.get_history()

    timestamps = pd.to_datetime(
        history["target_time"]
    )

    assert timestamps.is_monotonic_increasing

    assert timestamps.diff().dropna().eq(
        pd.Timedelta(minutes=10)
    ).all()


def test_energy_balance(simulator):

    for _ in range(100):
        simulator.step()

    history = simulator.get_history()

    np.testing.assert_allclose(
        history["always_normal_wh"],
        history["adaptive_consumption_wh"]
        + history["hypothetical_savings_wh"],
    )


def test_mode_decisions(simulator):

    for _ in range(100):
        simulator.step()

    history = simulator.get_history()

    expected = np.where(
        history["xgboost_prediction_wh"] > 100,
        "Eco",
        "Normal",
    )

    np.testing.assert_array_equal(
        history["mode"],
        expected,
    )


def test_empty_summary(simulator):

    summary = simulator.get_summary()

    assert summary["intervals"] == 0
    assert summary["eco_intervals"] == 0
    assert summary["hypothetical_savings_wh"] == 0


def test_summary_interval_counts(simulator):

    for _ in range(100):
        simulator.step()

    summary = simulator.get_summary()

    assert summary["intervals"] == 100

    assert (
        summary["normal_intervals"]
        + summary["eco_intervals"]
    ) == 100


def test_progress(simulator):

    simulator.step()

    assert simulator.progress == pytest.approx(
        1 / simulator.total_steps
    )


def test_replay_reproducibility(simulator):

    for _ in range(100):
        simulator.step()

    first_run = simulator.get_history().copy()

    simulator.reset()

    for _ in range(100):
        simulator.step()

    second_run = simulator.get_history().copy()

    pd.testing.assert_frame_equal(
        first_run.drop(columns=["processing_time_ms"]),
        second_run.drop(columns=["processing_time_ms"]),
    )