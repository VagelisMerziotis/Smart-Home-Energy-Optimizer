"""Confirm registered models load and match saved residual_predictions.csv.

The first Residual XGBoost test forecast (≈277.34 Wh) is a regression
anchor against the stored booster; a training-script change that alters
that value should fail this test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from config import (
    MODEL_SELECTION_PATH,
    RESIDUAL_PREDICTIONS_PATH,
    FINAL_CV_COMPARISON_PATH,
)

from simulator import EnergySimulator


MODEL_COLUMNS = {
    "Original XGBoost": "original_prediction_wh",
    "Optimized XGBoost": "optimized_prediction_wh",
    "Residual XGBoost": "residual_prediction_wh",
}


def test_final_model_selection():

    with open(MODEL_SELECTION_PATH, encoding="utf-8") as file:
        selection = json.load(file)

    assert selection["selected_model"] == "Residual XGBoost"


def test_selection_based_on_cv():

    with open(MODEL_SELECTION_PATH, encoding="utf-8") as file:
        selection = json.load(file)

    comparison = pd.read_csv(FINAL_CV_COMPARISON_PATH)

    candidates = comparison.loc[
        comparison["Model"] != "Persistence"
    ]

    best = candidates.sort_values(
        ["MAE", "RMSE"]
    ).iloc[0]

    assert selection["selected_model"] == best["Model"]

    assert selection["cv_mae_wh"] == pytest.approx(
        best["MAE"]
    )


def test_default_model():

    simulator = EnergySimulator()

    assert simulator.model_name == "Residual XGBoost"


@pytest.mark.parametrize(
    "model_name",
    list(MODEL_COLUMNS),
)
def test_model_loading(model_name):

    simulator = EnergySimulator(
        model_name=model_name
    )

    assert simulator.model_name == model_name

    assert simulator.total_steps == 3947


@pytest.mark.parametrize(
    "model_name",
    list(MODEL_COLUMNS),
)
def test_prediction_consistency(model_name):

    simulator = EnergySimulator(
        model_name=model_name
    )

    reference = pd.read_csv(
        RESIDUAL_PREDICTIONS_PATH
    )

    predictions = []

    for _ in range(20):

        record = simulator.step()

        predictions.append(
            record["xgboost_prediction_wh"]
        )

    expected = reference[
        MODEL_COLUMNS[model_name]
    ].iloc[:20].to_numpy()

    np.testing.assert_allclose(
        predictions,
        expected,
        rtol=1e-6,
        atol=1e-4,
    )


def test_residual_first_prediction():

    simulator = EnergySimulator(
        model_name="Residual XGBoost"
    )

    record = simulator.step()

    assert record[
        "xgboost_prediction_wh"
    ] == pytest.approx(
        277.3412,
        abs=0.001,
    )


def test_predictions_are_non_negative():

    simulator = EnergySimulator()

    for _ in range(100):

        record = simulator.step()

        prediction = record[
            "xgboost_prediction_wh"
        ]

        assert np.isfinite(prediction)
        assert prediction >= 0