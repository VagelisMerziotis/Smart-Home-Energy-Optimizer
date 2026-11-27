"""Load and validate the UCI Appliances Energy Prediction CSV.

The rest of the pipeline assumes a complete, regularly sampled series: no
missing values, unique timestamps, a 10-minute interval, and non-negative
appliance consumption. Failures here stop training and simulation early
rather than producing silent NaNs later.
"""

import pandas as pd

from config import (
    DATASET_PATH,
    DATE_COLUMN,
    TARGET_COLUMN,
    ENVIRONMENTAL_FEATURES,
    SAMPLING_INTERVAL_MINUTES,
)


def load_dataset(path=DATASET_PATH):
    """Return the sorted dataset after structural and sampling checks.

    Parameters
    ----------
    path : pathlib.Path
        Location of `energydata_complete.csv`.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing.
    ValueError
        If required columns, sampling, types, or value constraints fail.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        DATE_COLUMN,
        TARGET_COLUMN,
        *ENVIRONMENTAL_FEATURES,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="raise",
    )

    # Chronological order is required for lag features and the train/test cut.
    df = df.sort_values(DATE_COLUMN)
    df = df.reset_index(drop=True)

    if df.isna().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    if df[DATE_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    expected_interval = pd.Timedelta(
        minutes=SAMPLING_INTERVAL_MINUTES
    )

    intervals = df[DATE_COLUMN].diff().dropna()

    if not intervals.eq(expected_interval).all():
        raise ValueError(
            "Irregular sampling intervals detected."
        )

    numeric_columns = [
        TARGET_COLUMN,
        *ENVIRONMENTAL_FEATURES,
    ]

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column {col} is not numeric."
            )

    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError(
            "Negative energy consumption detected."
        )

    return df


def print_dataset_summary(df):
    """Print row counts, date range, and a short feature preview."""

    print("=" * 55)
    print("SMART HOME ENERGY OPTIMIZER")
    print("Dataset Summary")
    print("=" * 55)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Start date: {df[DATE_COLUMN].min()}")
    print(f"End date: {df[DATE_COLUMN].max()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(
        f"Duplicate timestamps: "
        f"{df[DATE_COLUMN].duplicated().sum()}"
    )
    print(
        f"Sampling interval: "
        f"{SAMPLING_INTERVAL_MINUTES} minutes"
    )

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nEnergy consumption statistics (Wh):")
    print(df[TARGET_COLUMN].describe().round(2))

    print("\nFirst five observations:")
    print(
        df[
            [
                DATE_COLUMN,
                TARGET_COLUMN,
                *ENVIRONMENTAL_FEATURES,
            ]
        ].head().to_string(index=False)
    )

    print("\nDataset validation: PASSED")
    print("=" * 55)


if __name__ == "__main__":
    dataset = load_dataset()
    print_dataset_summary(dataset)
