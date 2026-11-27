"""Unit tests for threshold decisions and the 30% × 20% Eco savings identity.

Eco savings of 12 Wh on 200 Wh actual follow from 200 * 0.30 * 0.20.
"""

import pytest

from policy_engine import PolicyEngine, EnergyMode


@pytest.mark.parametrize(
    "prediction, expected_mode",
    [
        (0, EnergyMode.NORMAL),
        (50, EnergyMode.NORMAL),
        (80, EnergyMode.NORMAL),
        (99.99, EnergyMode.NORMAL),
        (100, EnergyMode.NORMAL),
        (100.01, EnergyMode.ECO),
        (130, EnergyMode.ECO),
        (500, EnergyMode.ECO),
    ],
)
def test_policy_decisions(
    policy_engine,
    prediction,
    expected_mode,
):
    assert policy_engine.decide(prediction) == expected_mode


@pytest.mark.parametrize(
    "actual, mode, expected_savings",
    [
        (200, EnergyMode.NORMAL, 0),
        (200, EnergyMode.ECO, 12),
        (300, EnergyMode.ECO, 18),
        (100, EnergyMode.ECO, 6),
        (0, EnergyMode.ECO, 0),
    ],
)
def test_energy_savings(
    policy_engine,
    actual,
    mode,
    expected_savings,
):
    savings = policy_engine.calculate_savings(actual, mode)

    assert savings == pytest.approx(expected_savings)


@pytest.mark.parametrize(
    "actual, mode, expected_consumption",
    [
        (200, EnergyMode.NORMAL, 200),
        (200, EnergyMode.ECO, 188),
        (300, EnergyMode.ECO, 282),
    ],
)
def test_simulated_consumption(
    policy_engine,
    actual,
    mode,
    expected_consumption,
):
    simulated = policy_engine.simulate_consumption(actual, mode)

    assert simulated == pytest.approx(expected_consumption)


@pytest.mark.parametrize(
    "prediction",
    [-1, float("nan"), float("inf")],
)
def test_invalid_prediction(policy_engine, prediction):
    with pytest.raises(ValueError):
        policy_engine.decide(prediction)


@pytest.mark.parametrize(
    "threshold",
    [-1, float("nan"), float("inf")],
)
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        PolicyEngine(eco_threshold=threshold)


@pytest.mark.parametrize(
    "actual",
    [-1, float("nan"), float("inf")],
)
def test_invalid_consumption(policy_engine, actual):
    with pytest.raises(ValueError):
        policy_engine.calculate_savings(
            actual,
            EnergyMode.ECO,
        )