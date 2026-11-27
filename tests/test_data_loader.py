"""Checks that the UCI CSV matches the documented size, range, and sampling."""

import pandas as pd

from config import (
    DATE_COLUMN,
    TARGET_COLUMN,
    ENVIRONMENTAL_FEATURES,
)


def test_dataset_row_count(raw_dataset):
    assert len(raw_dataset) == 19735


def test_dataset_column_count(raw_dataset):
    assert len(raw_dataset.columns) == 29


def test_required_columns(raw_dataset):
    required = [
        DATE_COLUMN,
        TARGET_COLUMN,
        *ENVIRONMENTAL_FEATURES,
    ]

    assert set(required).issubset(raw_dataset.columns)


def test_no_missing_values(raw_dataset):
    assert raw_dataset.isna().sum().sum() == 0


def test_no_duplicate_timestamps(raw_dataset):
    assert not raw_dataset[DATE_COLUMN].duplicated().any()


def test_chronological_order(raw_dataset):
    assert raw_dataset[DATE_COLUMN].is_monotonic_increasing


def test_sampling_interval(raw_dataset):
    intervals = raw_dataset[DATE_COLUMN].diff().dropna()

    assert intervals.eq(pd.Timedelta(minutes=10)).all()


def test_non_negative_consumption(raw_dataset):
    assert (raw_dataset[TARGET_COLUMN] >= 0).all()


def test_dataset_date_range(raw_dataset):
    assert raw_dataset[DATE_COLUMN].min() == pd.Timestamp(
        "2016-01-11 17:00:00"
    )

    assert raw_dataset[DATE_COLUMN].max() == pd.Timestamp(
        "2016-05-27 18:00:00"
    )