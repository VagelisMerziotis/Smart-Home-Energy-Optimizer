"""Build forecast features and a leakage-safe chronological split.

Each row at decision time t uses only information available then: clock
features, current indoor/outdoor conditions, and the last three appliance
readings. The regression target is Appliances at t + 10 minutes. The
train/test cut is defined on the raw timeline, then applied to target
timestamps so a training label is never taken from the test period.
"""

import pandas as pd
from data_loader import load_dataset

from config import (
    DATE_COLUMN,
    TARGET_COLUMN,
    ENVIRONMENTAL_FEATURES,
    LAG_STEPS,
    FORECAST_STEPS,
    FORECAST_HORIZON_MINUTES,
    TRAIN_RATIO,
    ECO_PERCENTILE,
)


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    *ENVIRONMENTAL_FEATURES,
    *[f"appliances_lag_{lag}" for lag in LAG_STEPS],
]

TARGET_NAME = "target"
TARGET_TIME_COLUMN = "target_time"


def create_features(df):
    """Add time features, consumption lags, and the one-step-ahead target.

    Incomplete lag or target rows are dropped. The remaining horizon between
    `date` and `target_time` must equal FORECAST_HORIZON_MINUTES.
    """

    data = df.copy()

    # Fractional hour keeps 10-minute resolution instead of flooring to 0–23.
    data["hour"] = (
        data[DATE_COLUMN].dt.hour
        + data[DATE_COLUMN].dt.minute / 60
    )

    data["day_of_week"] = (
        data[DATE_COLUMN].dt.day_of_week
    )

    for lag in LAG_STEPS:
        data[f"appliances_lag_{lag}"] = (
            data[TARGET_COLUMN].shift(lag)
        )

    data[TARGET_NAME] = (
        data[TARGET_COLUMN].shift(-FORECAST_STEPS)
    )

    data[TARGET_TIME_COLUMN] = (
        data[DATE_COLUMN].shift(-FORECAST_STEPS)
    )

    selected_columns = [
        DATE_COLUMN,
        TARGET_TIME_COLUMN,
        *FEATURE_COLUMNS,
        TARGET_NAME,
    ]

    data = data[selected_columns]
    data = data.dropna().reset_index(drop=True)

    expected_horizon = pd.Timedelta(
        minutes=FORECAST_HORIZON_MINUTES
    )

    actual_horizon = (
        data[TARGET_TIME_COLUMN] - data[DATE_COLUMN]
    )

    if not actual_horizon.eq(expected_horizon).all():
        raise ValueError("Invalid forecast horizon.")

    return data


def chronological_split(data, raw_df):
    """Split featured rows so every training target occurs before the cutoff.

    The cutoff is the timestamp at TRAIN_RATIO of the raw series, not of the
    featured table. Using target_time (not decision time) prevents a training
    sample whose label sits inside the test window.
    """

    split_index = int(len(raw_df) * TRAIN_RATIO)
    cutoff_time = raw_df.iloc[split_index][DATE_COLUMN]

    train = data[
        data[TARGET_TIME_COLUMN] < cutoff_time
    ].copy()

    test = data[
        data[TARGET_TIME_COLUMN] >= cutoff_time
    ].copy()

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError("Empty training or test set.")

    if (
        train[TARGET_TIME_COLUMN].max()
        >= test[TARGET_TIME_COLUMN].min()
    ):
        raise ValueError("Train/test overlap detected.")

    return train, test, cutoff_time


def calculate_eco_threshold(raw_df, cutoff_time):
    """Return the Eco threshold from training-period consumption only."""

    training_values = raw_df.loc[
        raw_df[DATE_COLUMN] < cutoff_time,
        TARGET_COLUMN,
    ]

    threshold = training_values.quantile(
        ECO_PERCENTILE
    )

    return float(threshold)


def prepare_data():
    """Load, featurize, split, and compute the Eco threshold."""

    raw_df = load_dataset()
    featured_df = create_features(raw_df)
    train, test, cutoff_time = chronological_split(
        featured_df,
        raw_df,
    )
    eco_threshold = calculate_eco_threshold(
        raw_df,
        cutoff_time,
    )

    return train, test, eco_threshold


def print_preprocessing_summary(
    train,
    test,
    eco_threshold,
):
    """Print feature names, split sizes, and target date ranges."""

    print("=" * 60)
    print("SMART HOME ENERGY OPTIMIZER")
    print("Feature Engineering Summary")
    print("=" * 60)

    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    print("\nInput features:")

    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")

    print(f"\nTarget: {TARGET_NAME}")
    print(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON_MINUTES} minutes"
    )

    print(f"\nTraining samples: {len(train):,}")
    print(f"Test samples: {len(test):,}")
    print(f"Total samples: {len(train) + len(test):,}")

    print("\nTraining target period:")
    print(f"  From: {train[TARGET_TIME_COLUMN].min()}")
    print(f"  To:   {train[TARGET_TIME_COLUMN].max()}")

    print("\nTest target period:")
    print(f"  From: {test[TARGET_TIME_COLUMN].min()}")
    print(f"  To:   {test[TARGET_TIME_COLUMN].max()}")

    print(f"\nEco threshold: {eco_threshold:.2f} Wh")

    print("\nFirst five training samples:")

    preview_columns = [
        DATE_COLUMN,
        TARGET_TIME_COLUMN,
        "appliances_lag_2",
        "appliances_lag_1",
        "appliances_lag_0",
        TARGET_NAME,
    ]

    print(
        train[preview_columns]
        .head()
        .to_string(index=False)
    )

    print("\nPreprocessing validation: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    train, test, eco_threshold = prepare_data()

    print_preprocessing_summary(
        train,
        test,
        eco_threshold,
    )
