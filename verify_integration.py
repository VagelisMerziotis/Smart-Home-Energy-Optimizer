"""Check that simulation logs match residual predictions and energy tables.

Run after simulator.py. Assertions cover model identity, timestamps,
forecasts, Eco decisions, and the 30% × 20% savings identity.
"""

import json

import numpy as np
import pandas as pd

from config import (
    SIMULATION_LOG_PATH,
    SIMULATION_METRICS_PATH,
    RESIDUAL_PREDICTIONS_PATH,
    MODEL_SELECTION_PATH,
    ENERGY_POLICY_COMPARISON_PATH,
    CONTROLLABLE_LOAD_RATIO,
    ECO_REDUCTION_RATIO,
)


def main():
    """Load saved artifacts and fail fast on the first inconsistent check."""

    print("=" * 65)
    print("FINAL MODEL INTEGRATION VERIFICATION")
    print("=" * 65)

    with open(
        MODEL_SELECTION_PATH,
        encoding="utf-8",
    ) as file:

        selection = json.load(file)

    with open(
        SIMULATION_METRICS_PATH,
        encoding="utf-8",
    ) as file:

        summary = json.load(file)

    log = pd.read_csv(
        SIMULATION_LOG_PATH,
        parse_dates=[
            "decision_time",
            "target_time",
        ],
    )

    predictions = pd.read_csv(
        RESIDUAL_PREDICTIONS_PATH,
        parse_dates=[
            "decision_time",
            "target_time",
        ],
    )

    energy = pd.read_csv(
        ENERGY_POLICY_COMPARISON_PATH
    )

    selected_model = selection[
        "selected_model"
    ]

    prediction_columns = {
        "Original XGBoost": (
            "original_prediction_wh"
        ),
        "Optimized XGBoost": (
            "optimized_prediction_wh"
        ),
        "Residual XGBoost": (
            "residual_prediction_wh"
        ),
    }

    prediction_column = prediction_columns[
        selected_model
    ]

    expected = energy.loc[
        energy["Policy"]
        == f"Adaptive {selected_model}"
    ].iloc[0]

    # 1. Correct model

    assert (
        summary["model_name"]
        == selected_model
    )

    assert log["model_name"].eq(
        selected_model
    ).all()

    print("[PASS] Correct selected model")

    # 2. Correct number of intervals

    assert len(log) == len(predictions)

    assert len(log) == expected["Intervals"]

    assert summary["intervals"] == len(log)

    print("[PASS] Correct interval count")

    # 3. Correct chronological alignment

    assert log["decision_time"].equals(
        predictions["decision_time"]
    )

    assert log["target_time"].equals(
        predictions["target_time"]
    )

    assert (
        log["target_time"]
        - log["decision_time"]
    ).eq(
        pd.Timedelta(minutes=10)
    ).all()

    print("[PASS] Chronological alignment")

    # 4. Correct actual consumption

    np.testing.assert_allclose(
        log["actual_wh"],
        predictions["actual_wh"],
        rtol=0,
        atol=1e-8,
    )

    print("[PASS] Actual consumption")

    # 5. Correct predictions

    np.testing.assert_allclose(
        log["xgboost_prediction_wh"],
        predictions[prediction_column],
        rtol=1e-6,
        atol=1e-4,
    )

    max_difference = np.max(
        np.abs(
            log["xgboost_prediction_wh"]
            - predictions[prediction_column]
        )
    )

    print("[PASS] Prediction consistency")

    print(
        f"       Maximum difference: "
        f"{max_difference:.8f} Wh"
    )

    # 6. Correct energy mode

    expected_modes = np.where(
        log["xgboost_prediction_wh"]
        > summary["eco_threshold_wh"],
        "Eco",
        "Normal",
    )

    np.testing.assert_array_equal(
        log["mode"].to_numpy(),
        expected_modes,
    )

    print("[PASS] Eco policy decisions")

    # 7. Correct interval savings

    expected_savings = np.where(
        log["mode"] == "Eco",
        log["actual_wh"]
        * CONTROLLABLE_LOAD_RATIO
        * ECO_REDUCTION_RATIO,
        0.0,
    )

    np.testing.assert_allclose(
        log["hypothetical_savings_wh"],
        expected_savings,
        rtol=0,
        atol=1e-8,
    )

    print("[PASS] Interval energy savings")

    # 8. Energy balance

    np.testing.assert_allclose(
        log["always_normal_wh"],
        log["actual_wh"],
        rtol=0,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        log["adaptive_consumption_wh"]
        + log["hypothetical_savings_wh"],
        log["actual_wh"],
        rtol=0,
        atol=1e-8,
    )

    print("[PASS] Energy balance")

    # 9. Correct mode counts

    assert (
        summary["eco_intervals"]
        == expected["Eco_intervals"]
        == int(log["mode"].eq("Eco").sum())
    )

    assert (
        summary["normal_intervals"]
        == expected["Normal_intervals"]
        == int(log["mode"].eq("Normal").sum())
    )

    print("[PASS] Normal/Eco interval counts")

    # 10. Correct total energy metrics

    np.testing.assert_allclose(
        summary["always_normal_wh"],
        log["actual_wh"].sum(),
        rtol=0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        summary["adaptive_consumption_wh"],
        expected["Consumption_Wh"],
        rtol=0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        summary["hypothetical_savings_wh"],
        expected["Savings_Wh"],
        rtol=0,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        summary["hypothetical_savings_percent"],
        expected["Savings_percent"],
        rtol=0,
        atol=1e-6,
    )

    print("[PASS] Total energy metrics")

    # 11. Core processing performance

    assert np.isfinite(
        log["processing_time_ms"]
    ).all()

    assert (
        log["processing_time_ms"] >= 0
    ).all()

    assert (
        log["processing_time_ms"].max()
        <= 1000
    )

    print("[PASS] Core processing below 1 second")

    print("\n" + "=" * 65)
    print("ALL INTEGRATION CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
