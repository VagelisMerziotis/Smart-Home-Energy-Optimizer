"""Tune a direct XGBoost regressor with time-series cross-validation.

RandomizedSearchCV uses only the training split. sklearn scorers are
defined as negative error, so CV tables negate those values back to MAE
and RMSE in Wh. Test-set numbers are computed after the search finishes.
"""

import json
import time

import numpy as np
import pandas as pd
import xgboost

from xgboost import XGBRegressor

from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)

from config import (
    DATE_COLUMN,
    RANDOM_STATE,
    XGBOOST_PARAMS,
    MODELS_DIR,
    RESULTS_DIR,
    MODEL_PATH,
    METADATA_PATH,
    FORECAST_HORIZON_MINUTES,
    TUNING_ITERATIONS,
    TUNING_CV_SPLITS,
    TUNING_N_JOBS,
    OPTIMIZED_MODEL_PATH,
    OPTIMIZED_METADATA_PATH,
    TUNING_RESULTS_PATH,
    BEST_PARAMETERS_PATH,
    OPTIMIZED_PREDICTIONS_PATH,
    OPTIMIZED_METRICS_PATH,
    MODEL_COMPARISON_PATH,
)

from preprocessing import (
    prepare_data,
    FEATURE_COLUMNS,
    TARGET_NAME,
    TARGET_TIME_COLUMN,
)


# --------------------------------------------------
# Hyperparameter search space
# --------------------------------------------------

PARAM_DISTRIBUTIONS = {
    "objective": [
        "reg:squarederror",
        "reg:absoluteerror",
        "reg:pseudohubererror",
    ],

    "n_estimators": [
        100, 200, 300, 500, 800,
    ],

    "max_depth": [
        2, 3, 4, 5, 6, 8,
    ],

    "learning_rate": [
        0.01, 0.03, 0.05, 0.1,
    ],

    "subsample": [
        0.7, 0.8, 0.9, 1.0,
    ],

    "colsample_bytree": [
        0.7, 0.8, 1.0,
    ],

    "min_child_weight": [
        1, 3, 5, 10,
    ],

    "gamma": [
        0, 0.1, 1, 5,
    ],

    "reg_alpha": [
        0, 0.1, 1, 5,
    ],

    "reg_lambda": [
        0.1, 1, 5, 10,
    ],
}


# --------------------------------------------------
# Prediction utilities
# --------------------------------------------------

def clip_predictions(predictions):
    """Map negative Wh forecasts to zero."""
    """Prevent negative energy predictions."""

    return np.maximum(
        np.asarray(predictions),
        0.0,
    )


def clipped_mae(y_true, y_pred):
    """MAE after clipping predictions to non-negative Wh."""
    return mean_absolute_error(
        y_true,
        clip_predictions(y_pred),
    )


def clipped_rmse(y_true, y_pred):
    """RMSE after clipping predictions to non-negative Wh."""
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                clip_predictions(y_pred),
            )
        )
    )


SCORING = {
    "MAE": make_scorer(
        clipped_mae,
        greater_is_better=False,
    ),
    "RMSE": make_scorer(
        clipped_rmse,
        greater_is_better=False,
    ),
}


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def evaluate_predictions(
    y_true,
    y_pred,
    threshold,
):
    """Evaluate regression and Eco detection."""

    y_true = np.asarray(y_true)
    y_pred = clip_predictions(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    # Actual and predicted threshold exceedance
    actual_high = (
        y_true > threshold
    ).astype(int)

    predicted_high = (
        y_pred > threshold
    ).astype(int)

    precision = precision_score(
        actual_high,
        predicted_high,
        zero_division=0,
    )

    recall = recall_score(
        actual_high,
        predicted_high,
        zero_division=0,
    )

    f1 = f1_score(
        actual_high,
        predicted_high,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        actual_high,
        predicted_high,
        labels=[0, 1],
    ).ravel()

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# --------------------------------------------------
# Time series validation
# --------------------------------------------------

def create_time_series_cv(train):
    """Create and validate chronological folds."""

    cv = TimeSeriesSplit(
        n_splits=TUNING_CV_SPLITS,
        gap=0,
    )

    print("\nTime Series Cross-Validation")
    print("-" * 65)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(train),
        start=1,
    ):

        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]

        train_end = fold_train[
            TARGET_TIME_COLUMN
        ].max()

        val_start = fold_val[
            TARGET_TIME_COLUMN
        ].min()

        first_decision = fold_val[
            DATE_COLUMN
        ].min()

        if train_end >= val_start:
            raise ValueError(
                "Temporal leakage detected."
            )

        if train_end > first_decision:
            raise ValueError(
                "Training label unavailable "
                "at validation decision time."
            )

        print(f"\nFold {fold}")
        print(
            f"  Training samples: "
            f"{len(train_idx):,}"
        )
        print(
            f"  Validation samples: "
            f"{len(val_idx):,}"
        )
        print(f"  Training end: {train_end}")
        print(f"  Validation start: {val_start}")

    return cv


# --------------------------------------------------
# Original model validation
# --------------------------------------------------

def evaluate_original_cv(
    X_train,
    y_train,
    cv,
):
    """Evaluate the original configuration on CV."""

    params = {
        **XGBOOST_PARAMS,
        "n_jobs": 1,
    }

    model = XGBRegressor(**params)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=TUNING_N_JOBS,
        error_score="raise",
    )

    return {
        "MAE": float(
            -scores["test_MAE"].mean()
        ),
        "RMSE": float(
            -scores["test_RMSE"].mean()
        ),
    }


def evaluate_persistence_cv(
    X_train,
    y_train,
    cv,
):
    """Evaluate persistence on the same folds."""

    mae_scores = []
    rmse_scores = []

    for _, val_idx in cv.split(X_train):

        actual = y_train.iloc[val_idx]

        predicted = X_train.iloc[val_idx][
            "appliances_lag_0"
        ]

        mae_scores.append(
            clipped_mae(actual, predicted)
        )

        rmse_scores.append(
            clipped_rmse(actual, predicted)
        )

    return {
        "MAE": float(np.mean(mae_scores)),
        "RMSE": float(np.mean(rmse_scores)),
    }


# --------------------------------------------------
# Hyperparameter optimization
# --------------------------------------------------

def tune_xgboost(
    X_train,
    y_train,
    cv,
):
    """Optimize XGBoost using training data only."""

    estimator = XGBRegressor(
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=TUNING_ITERATIONS,
        scoring=SCORING,
        refit="MAE",
        cv=cv,
        n_jobs=TUNING_N_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True,
        error_score="raise",
    )

    start_time = time.perf_counter()

    search.fit(
        X_train,
        y_train,
    )

    elapsed = (
        time.perf_counter() - start_time
    )

    return search, elapsed


# --------------------------------------------------
# Save cross-validation results
# --------------------------------------------------

def save_tuning_results(search):
    """Write all search rows ranked by CV MAE, with errors as positive Wh."""

    results = pd.DataFrame(
        search.cv_results_
    )

    # Convert negative scores to positive errors
    for dataset in ["train", "test"]:

        for metric in ["MAE", "RMSE"]:

            column = (
                f"mean_{dataset}_{metric}"
            )

            results[column] = (
                -results[column]
            )

    results = results.sort_values(
        "rank_test_MAE"
    )

    results.to_csv(
        TUNING_RESULTS_PATH,
        index=False,
    )

    print("\nTop 10 configurations by CV MAE:")
    print("-" * 65)

    columns = [
        "rank_test_MAE",
        "param_objective",
        "param_n_estimators",
        "param_max_depth",
        "param_learning_rate",
        "mean_test_MAE",
        "mean_test_RMSE",
    ]

    print(
        results[columns]
        .head(10)
        .round(4)
        .to_string(index=False)
    )


# --------------------------------------------------
# Save optimized model and metadata
# --------------------------------------------------

def save_optimized_model(
    search,
    train,
    test,
    threshold,
    tuning_time,
):
    """Persist the best booster, metadata, and the winning hyperparameter set."""

    model = search.best_estimator_

    model.save_model(
        str(OPTIMIZED_MODEL_PATH)
    )

    best_index = search.best_index_

    best_cv_mae = float(
        -search.cv_results_[
            "mean_test_MAE"
        ][best_index]
    )

    best_cv_rmse = float(
        -search.cv_results_[
            "mean_test_RMSE"
        ][best_index]
    )

    best_parameters = {
        "best_parameters": search.best_params_,
        "selection_metric": "MAE",
        "cv_mae_wh": best_cv_mae,
        "cv_rmse_wh": best_cv_rmse,
        "cv_splits": TUNING_CV_SPLITS,
        "search_iterations": TUNING_ITERATIONS,
        "tuning_time_seconds": tuning_time,
    }

    with open(
        BEST_PARAMETERS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_parameters,
            file,
            indent=4,
        )

    metadata = {
        "model": "Optimized XGBRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "forecast_horizon_minutes": (
            FORECAST_HORIZON_MINUTES
        ),
        "eco_threshold_wh": float(threshold),
        "training_samples": len(train),
        "test_samples": len(test),
        "train_start": str(
            train[TARGET_TIME_COLUMN].min()
        ),
        "train_end": str(
            train[TARGET_TIME_COLUMN].max()
        ),
        "test_start": str(
            test[TARGET_TIME_COLUMN].min()
        ),
        "test_end": str(
            test[TARGET_TIME_COLUMN].max()
        ),
        "model_parameters": {
            **search.best_params_,
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
            "n_jobs": 1,
        },
        "prediction_type": "direct",
        "cv_mae_wh": best_cv_mae,
        "cv_rmse_wh": best_cv_rmse,
        "xgboost_version": xgboost.__version__,
    }

    with open(
        OPTIMIZED_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


# --------------------------------------------------
# Print evaluation comparison
# --------------------------------------------------

def print_comparison(
    comparison,
    cv_comparison,
):
    """Print CV and test tables plus error reduction versus baselines."""

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 80)

    print(
        cv_comparison.round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("TEST SET MODEL COMPARISON")
    print("=" * 80)

    print(
        comparison.round(4)
        .to_string(index=False)
    )

    print("\nError reduction:")

    baseline = comparison.iloc[0]
    original = comparison.iloc[1]
    optimized = comparison.iloc[2]

    for metric in ["MAE", "RMSE"]:

        vs_baseline = (
            (baseline[metric] - optimized[metric])
            / baseline[metric]
            * 100
        )

        vs_original = (
            (original[metric] - optimized[metric])
            / original[metric]
            * 100
        )

        print(
            f"\n{metric}:"
        )

        print(
            f"  Optimized vs Persistence: "
            f"{vs_baseline:.2f}%"
        )

        print(
            f"  Optimized vs Original: "
            f"{vs_original:.2f}%"
        )


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SMART HOME ENERGY OPTIMIZER")
    print("XGBoost Hyperparameter Optimization")
    print("=" * 80)

    print(
        f"XGBoost version: "
        f"{xgboost.__version__}"
    )

    # 1. Prepare training and test data
    train, test, threshold = prepare_data()

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_NAME]

    print(
        f"Training samples: {len(train):,}"
    )

    print(
        f"Test samples: {len(test):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Search iterations: "
        f"{TUNING_ITERATIONS}"
    )

    print(
        f"Total search fits: "
        f"{TUNING_ITERATIONS * TUNING_CV_SPLITS}"
    )

    # 2. Create chronological validation folds
    cv = create_time_series_cv(train)

    # 3. Evaluate reference models on CV
    print("\nEvaluating persistence on CV...")

    persistence_cv = evaluate_persistence_cv(
        X_train,
        y_train,
        cv,
    )

    print("\nEvaluating original XGBoost on CV...")

    original_cv = evaluate_original_cv(
        X_train,
        y_train,
        cv,
    )

    # 4. Optimize using training data only
    print("\nStarting hyperparameter search...")

    search, tuning_time = tune_xgboost(
        X_train,
        y_train,
        cv,
    )

    print("\nHyperparameter search completed.")

    print(
        f"Tuning time: "
        f"{tuning_time:.2f} seconds"
    )

    print("\nBest parameters:")

    print(
        json.dumps(
            search.best_params_,
            indent=4,
        )
    )

    best_index = search.best_index_

    optimized_cv = {
        "MAE": float(
            -search.cv_results_[
                "mean_test_MAE"
            ][best_index]
        ),
        "RMSE": float(
            -search.cv_results_[
                "mean_test_RMSE"
            ][best_index]
        ),
    }

    cv_comparison = pd.DataFrame([
        {
            "Model": "Persistence",
            **persistence_cv,
        },
        {
            "Model": "Original XGBoost",
            **original_cv,
        },
        {
            "Model": "Optimized XGBoost",
            **optimized_cv,
        },
    ])

    cv_comparison.to_csv(
        RESULTS_DIR / "cv_model_comparison.csv",
        index=False,
    )

    save_tuning_results(search)

    save_optimized_model(
        search,
        train,
        test,
        threshold,
        tuning_time,
    )

    # 5. Evaluate on test data only after tuning
    print("\nEvaluating models on test set...")

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_NAME]

    with open(
        METADATA_PATH,
        encoding="utf-8",
    ) as file:

        original_metadata = json.load(file)

    if (
        original_metadata["feature_columns"]
        != FEATURE_COLUMNS
    ):
        raise ValueError(
            "Original model feature mismatch."
        )

    if (
        original_metadata["training_samples"]
        != len(train)
        or original_metadata["test_samples"]
        != len(test)
        or original_metadata["train_end"]
        != str(train[TARGET_TIME_COLUMN].max())
        or original_metadata["test_start"]
        != str(test[TARGET_TIME_COLUMN].min())
    ):
        raise ValueError(
            "Original model split mismatch."
        )

    original_model = XGBRegressor()
    original_model.load_model(str(MODEL_PATH))

    persistence_predictions = (
        X_test["appliances_lag_0"]
        .to_numpy()
    )

    original_predictions = clip_predictions(
        original_model.predict(X_test)
    )

    optimized_predictions = clip_predictions(
        search.best_estimator_.predict(X_test)
    )

    # 6. Evaluate all models
    all_metrics = {
        "Persistence": evaluate_predictions(
            y_test,
            persistence_predictions,
            threshold,
        ),
        "Original XGBoost": evaluate_predictions(
            y_test,
            original_predictions,
            threshold,
        ),
        "Optimized XGBoost": evaluate_predictions(
            y_test,
            optimized_predictions,
            threshold,
        ),
    }

    comparison = pd.DataFrame([
        {
            "Model": model_name,
            **metrics,
        }
        for model_name, metrics
        in all_metrics.items()
    ])

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    # 7. Save predictions
    predictions = pd.DataFrame({
        "decision_time": test[DATE_COLUMN],
        "target_time": test[TARGET_TIME_COLUMN],
        "actual_wh": y_test.to_numpy(),
        "baseline_prediction_wh": (
            persistence_predictions
        ),
        "original_prediction_wh": (
            original_predictions
        ),
        "optimized_prediction_wh": (
            optimized_predictions
        ),
    })

    predictions.to_csv(
        OPTIMIZED_PREDICTIONS_PATH,
        index=False,
    )

    # 8. Save all metrics
    output_metrics = {
        "cross_validation": {
            "Persistence": persistence_cv,
            "Original XGBoost": original_cv,
            "Optimized XGBoost": optimized_cv,
        },
        "test_metrics": all_metrics,
        "eco_threshold_wh": float(threshold),
        "tuning_time_seconds": tuning_time,
        "refit_time_seconds": float(
            search.refit_time_
        ),
        "best_parameters": search.best_params_,
    }

    with open(
        OPTIMIZED_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_metrics,
            file,
            indent=4,
        )

    print_comparison(
        comparison,
        cv_comparison,
    )

    print("\nSaved artifacts:")

    print(f"  Model: {OPTIMIZED_MODEL_PATH}")
    print(f"  Metadata: {OPTIMIZED_METADATA_PATH}")
    print(f"  Tuning results: {TUNING_RESULTS_PATH}")
    print(f"  Best parameters: {BEST_PARAMETERS_PATH}")
    print(f"  Predictions: {OPTIMIZED_PREDICTIONS_PATH}")
    print(f"  Metrics: {OPTIMIZED_METRICS_PATH}")
    print(f"  Comparison: {MODEL_COMPARISON_PATH}")

    print("\nOptimization: COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
