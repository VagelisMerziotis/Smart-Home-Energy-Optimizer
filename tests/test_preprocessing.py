"""Lock the feature set, chronological split, lag alignment, and Eco cutoff."""

import numpy as np
import pandas as pd
import pytest

from config import DATE_COLUMN, TARGET_COLUMN

from preprocessing import (
    FEATURE_COLUMNS,
    TARGET_NAME,
    TARGET_TIME_COLUMN,
    create_features,
)


def test_feature_count():
    assert len(FEATURE_COLUMNS) == 10


def test_training_sample_count(train_data):
    assert len(train_data) == 15785


def test_test_sample_count(test_data):
    assert len(test_data) == 3947


def test_no_temporal_overlap(train_data, test_data):
    assert (
        train_data[TARGET_TIME_COLUMN].max()
        < test_data[TARGET_TIME_COLUMN].min()
    )


def test_first_test_timestamp(test_data):
    assert test_data[TARGET_TIME_COLUMN].min() == pd.Timestamp(
        "2016-04-30 08:20:00"
    )


def test_forecast_horizon(train_data, test_data):
    data = pd.concat([train_data, test_data])

    horizon = data[TARGET_TIME_COLUMN] - data[DATE_COLUMN]

    assert horizon.eq(pd.Timedelta(minutes=10)).all()


def test_no_missing_features(train_data, test_data):
    assert not train_data[FEATURE_COLUMNS].isna().any().any()
    assert not test_data[FEATURE_COLUMNS].isna().any().any()


def test_target_not_in_features():
    assert TARGET_NAME not in FEATURE_COLUMNS
    assert TARGET_TIME_COLUMN not in FEATURE_COLUMNS


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_lag_alignment(raw_dataset, lag):
    featured = create_features(raw_dataset)

    expected = raw_dataset[TARGET_COLUMN].shift(lag)

    reference = pd.Series(
        expected.to_numpy(),
        index=raw_dataset[DATE_COLUMN],
    )

    expected_values = reference.reindex(
        featured[DATE_COLUMN]
    ).to_numpy()

    np.testing.assert_allclose(
        featured[f"appliances_lag_{lag}"].to_numpy(),
        expected_values,
    )


def test_target_alignment(raw_dataset):
    featured = create_features(raw_dataset)

    reference = raw_dataset.set_index(DATE_COLUMN)[TARGET_COLUMN]

    expected = reference.reindex(
        featured[TARGET_TIME_COLUMN]
    ).to_numpy()

    np.testing.assert_allclose(
        featured[TARGET_NAME].to_numpy(),
        expected,
    )


def test_eco_threshold(eco_threshold):
    assert eco_threshold == pytest.approx(100.0)


def test_threshold_uses_training_only(
    raw_dataset,
    test_data,
    eco_threshold,
):
    cutoff = test_data[TARGET_TIME_COLUMN].min()

    training_values = raw_dataset.loc[
        raw_dataset[DATE_COLUMN] < cutoff,
        TARGET_COLUMN,
    ]

    expected = training_values.quantile(0.75)

    assert eco_threshold == pytest.approx(expected)