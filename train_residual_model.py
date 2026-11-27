"""Tune Residual XGBoost, pick the final model from CV MAE, and score energy.

The sklearn-compatible ResidualXGBRegressor trains on y(t+1) - y(t) and
reconstructs absolute Wh at predict time. Model selection ignores test MAE
so the held-out numbers remain a report, not a selection criterion.
"""

import json
import time

import numpy as np
import pandas as pd
import xgboost

from xgboost import XGBRegressor

from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
)

from sklearn.model_selection import (
    RandomizedSearchCV,
)

from sklearn.utils.validation import (
    check_is_fitted,
)

from config import (
    DATE_COLUMN,
    RANDOM_STATE,
    RESULTS_DIR,
    MODELS_DIR,
    MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    OPTIMIZED_METRICS_PATH,
    OPTIMIZED_METADATA_PATH,
    FORECAST_HORIZON_MINUTES,
    TUNING_ITERATIONS,
    TUNING_CV_SPLITS,
    TUNING_N_JOBS,
    CONTROLLABLE_LOAD_RATIO,
    ECO_REDUCTION_RATIO,
    RESIDUAL_MODEL_PATH,
    RESIDUAL_METADATA_PATH,
    RESIDUAL_TUNING_RESULTS_PATH,
    RESIDUAL_BEST_PARAMETERS_PATH,
    RESIDUAL_PREDICTIONS_PATH,
    RESIDUAL_METRICS_PATH,
    FINAL_MODEL_COMPARISON_PATH,
    FINAL_CV_COMPARISON_PATH,
    MODEL_SELECTION_PATH,
    ENERGY_POLICY_COMPARISON_PATH,
)

from preprocessing import (
    prepare_data,
    FEATURE_COLUMNS,
    TARGET_NAME,
    TARGET_TIME_COLUMN,
)

from tune_model import (
    PARAM_DISTRIBUTIONS,
    SCORING,
    clip_predictions,
    evaluate_predictions,
    create_time_series_cv,
)


# --------------------------------------------------
# Residual XGBoost estimator
# --------------------------------------------------

class ResidualXGBRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """XGBoost wrapper that learns the change from the last observed load."""


    def __init__(
        self,
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    ):

        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method

    def fit(self, X, y):
        """Fit an inner XGBRegressor on residual targets."""


        # Last known consumption at decision time
        baseline = X[
            "appliances_lag_0"
        ].to_numpy(dtype=float)

        actual = np.asarray(
            y,
            dtype=float,
        )

        # Residual target:
        # future consumption - current consumption
        residual_target = actual - baseline

        self.model_ = XGBRegressor(
            **self.get_params(deep=False),
            verbosity=0,
        )

        self.model_.fit(
            X,
            residual_target,
        )

        self.n_features_in_ = X.shape[1]

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        return self

    def predict(self, X):
        """Return reconstructed, clipped absolute consumption."""


        check_is_fitted(
            self,
            "model_",
        )

        baseline = X[
            "appliances_lag_0"
        ].to_numpy(dtype=float)

        predicted_residual = (
            self.model_.predict(X)
        )

        # Reconstruct absolute consumption
        predictions = (
            baseline + predicted_residual
        )

        return clip_predictions(predictions)

    def predict_residual(self, X):
        """Return the raw residual (change) without adding the baseline."""


        check_is_fitted(
            self,
            "model_",
        )

        return self.model_.predict(X)


# --------------------------------------------------
# Residual hyperparameter optimization
# --------------------------------------------------

def tune_residual_model(
    X_train,
    y_train,
    cv,
):
    """Random-search ResidualXGBRegressor on the same folds as direct models."""

    estimator = ResidualXGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=TUNING_ITERATIONS,
        scoring=SCORING,
        refit="MAE",
        cv=cv,
        n_jobs=TUNING_N_JOBS,
        verbose=1,
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
# Save tuning results
# --------------------------------------------------

def save_tuning_results(search):
    """Write residual search rows with MAE/RMSE converted back to positive Wh."""

    results = pd.DataFrame(
        search.cv_results_
    )

    # Convert negative scorer values
    # into positive error values
    for column in results.columns:

        is_error_score = (
            column.endswith("_MAE")
            or column.endswith("_RMSE")
        )

        is_mean_or_fold = (
            column.startswith("mean_")
            or column.startswith("split")
        )

        if is_error_score and is_mean_or_fold:

            results[column] = (
                -results[column]
            )

    results = results.sort_values(
        "rank_test_MAE"
    )

    results.to_csv(
        RESIDUAL_TUNING_RESULTS_PATH,
        index=False,
    )

    columns = [
        "rank_test_MAE",
        "param_objective",
        "param_n_estimators",
        "param_max_depth",
        "param_learning_rate",
        "mean_test_MAE",
        "mean_test_RMSE",
    ]

    print("\nTop 10 residual configurations:")
    print("-" * 80)

    print(
        results[columns]
        .head(10)
        .round(4)
        .to_string(index=False)
    )


# --------------------------------------------------
# Save model and metadata
# --------------------------------------------------

def save_residual_model(
    search,
    train,
    test,
    threshold,
    tuning_time,
    cv_metrics,
):
    """Save the inner booster plus reconstruction metadata."""

    estimator = search.best_estimator_

    estimator.model_.save_model(
        str(RESIDUAL_MODEL_PATH)
    )

    metadata = {
        "model": "Residual XGBRegressor",
        "prediction_type": "residual",
        "residual_definition": (
            "target - appliances_lag_0"
        ),
        "reconstruction": (
            "max(0, appliances_lag_0 "
            "+ predicted_residual)"
        ),
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
        "model_parameters": (
            estimator.get_params(deep=False)
        ),
        "cv_mae_wh": cv_metrics["MAE"],
        "cv_rmse_wh": cv_metrics["RMSE"],
        "xgboost_version": xgboost.__version__,
    }

    with open(
        RESIDUAL_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    parameters = {
        "best_parameters": search.best_params_,
        "selection_metric": "MAE",
        "cv_mae_wh": cv_metrics["MAE"],
        "cv_rmse_wh": cv_metrics["RMSE"],
        "cv_splits": TUNING_CV_SPLITS,
        "search_iterations": TUNING_ITERATIONS,
        "tuning_time_seconds": tuning_time,
    }

    with open(
        RESIDUAL_BEST_PARAMETERS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            parameters,
            file,
            indent=4,
        )


# --------------------------------------------------
# Energy policy comparison
# --------------------------------------------------

def evaluate_energy_policy(
    actual,
    predicted,
    threshold,
    policy_name,
):
    """Apply the same Eco rule as PolicyEngine to a full prediction vector."""

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    eco_mask = predicted > threshold

    savings = np.where(
        eco_mask,
        actual
        * CONTROLLABLE_LOAD_RATIO
        * ECO_REDUCTION_RATIO,
        0.0,
    )

    total_actual = float(actual.sum())
    total_savings = float(savings.sum())

    total_adaptive = (
        total_actual - total_savings
    )

    return {
        "Policy": policy_name,
        "Intervals": len(actual),
        "Normal_intervals": int(
            (~eco_mask).sum()
        ),
        "Eco_intervals": int(
            eco_mask.sum()
        ),
        "Eco_percent": float(
            eco_mask.mean() * 100
        ),
        "Consumption_Wh": total_adaptive,
        "Savings_Wh": total_savings,
        "Savings_percent": (
            total_savings / total_actual * 100
            if total_actual > 0
            else 0.0
        ),
    }


# --------------------------------------------------
# Select final XGBoost model
# --------------------------------------------------

def select_final_model(cv_comparison):
    """Choose the XGBoost variant with lowest CV MAE (RMSE as tie-breaker)."""

    # Persistence remains the reference method.
    # Selection is among XGBoost variants only.

    candidates = cv_comparison[
        cv_comparison["Model"] != "Persistence"
    ]

    ranked = candidates.sort_values(
        ["MAE", "RMSE"],
        ascending=True,
    )

    winner = ranked.iloc[0]

    baseline = cv_comparison[
        cv_comparison["Model"] == "Persistence"
    ].iloc[0]

    selection = {
        "selected_model": winner["Model"],
        "selection_criterion": (
            "Minimum mean time-series CV MAE"
        ),
        "tie_breaker": "CV RMSE",
        "cv_mae_wh": float(winner["MAE"]),
        "cv_rmse_wh": float(winner["RMSE"]),
        "persistence_cv_mae_wh": float(
            baseline["MAE"]
        ),
        "beats_persistence_cv_mae": bool(
            winner["MAE"] < baseline["MAE"]
        ),
        "test_metrics_used_for_selection": False,
    }

    with open(
        MODEL_SELECTION_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            selection,
            file,
            indent=4,
        )

    return selection


# --------------------------------------------------
# Validate existing optimized artifacts
# --------------------------------------------------

def validate_existing_artifacts(
    train,
    test,
    threshold,
):
    """Confirm optimized-model metadata still matches the current data split."""

    with open(
        OPTIMIZED_METADATA_PATH,
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if (
        metadata["feature_columns"]
        != FEATURE_COLUMNS
    ):
        raise ValueError(
            "Feature mismatch."
        )

    if (
        metadata["training_samples"]
        != len(train)
        or metadata["test_samples"]
        != len(test)
    ):
        raise ValueError(
            "Dataset size mismatch."
        )

    if (
        metadata["train_end"]
        != str(train[TARGET_TIME_COLUMN].max())
        or metadata["test_start"]
        != str(test[TARGET_TIME_COLUMN].min())
    ):
        raise ValueError(
            "Chronological split mismatch."
        )

    if not np.isclose(
        metadata["eco_threshold_wh"],
        threshold,
    ):
        raise ValueError(
            "Eco threshold mismatch."
        )


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():
    """Tune residual model, select from CV, then write test and energy tables."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SMART HOME ENERGY OPTIMIZER")
    print("Residual XGBoost Optimization")
    print("=" * 80)

    # 1. Prepare data

    train, test, threshold = prepare_data()

    validate_existing_artifacts(
        train,
        test,
        threshold,
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_NAME]

    residual_target = (
        y_train
        - X_train["appliances_lag_0"]
    )

    print(
        f"Training samples: {len(train):,}"
    )

    print(
        f"Test samples: {len(test):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print("\nResidual target statistics:")

    print(
        residual_target.describe()
        .round(4)
        .to_string()
    )

    # 2. Create chronological folds

    cv = create_time_series_cv(train)

    # 3. Run optimization

    print("\nStarting residual optimization...")

    search, tuning_time = tune_residual_model(
        X_train,
        y_train,
        cv,
    )

    print("\nResidual optimization completed.")

    print(
        f"Tuning time: {tuning_time:.2f} sec"
    )

    print("\nBest parameters:")

    print(
        json.dumps(
            search.best_params_,
            indent=4,
        )
    )

    best_index = search.best_index_

    residual_cv = {
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

    save_tuning_results(search)

    save_residual_model(
        search,
        train,
        test,
        threshold,
        tuning_time,
        residual_cv,
    )

    # 4. Load previous CV results

    with open(
        OPTIMIZED_METRICS_PATH,
        encoding="utf-8",
    ) as file:

        previous_metrics = json.load(file)

    all_cv_metrics = {
        **previous_metrics["cross_validation"],
        "Residual XGBoost": residual_cv,
    }

    cv_comparison = pd.DataFrame([
        {
            "Model": name,
            **metrics,
        }
        for name, metrics
        in all_cv_metrics.items()
    ])

    cv_comparison.to_csv(
        FINAL_CV_COMPARISON_PATH,
        index=False,
    )

    # 5. Select model before evaluating test set

    selection = select_final_model(
        cv_comparison
    )

    print("\n" + "=" * 80)
    print("FINAL CROSS-VALIDATION COMPARISON")
    print("=" * 80)

    print(
        cv_comparison.round(4)
        .to_string(index=False)
    )

    print("\nModel selected using CV only:")

    print(
        json.dumps(
            selection,
            indent=4,
        )
    )

    # 6. Test set evaluation

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_NAME]

    original_model = XGBRegressor()

    original_model.load_model(
        str(MODEL_PATH)
    )

    optimized_model = XGBRegressor()

    optimized_model.load_model(
        str(OPTIMIZED_MODEL_PATH)
    )

    persistence_predictions = (
        X_test["appliances_lag_0"]
        .to_numpy()
    )

    original_predictions = clip_predictions(
        original_model.predict(X_test)
    )

    optimized_predictions = clip_predictions(
        optimized_model.predict(X_test)
    )

    residual_predictions = (
        search.best_estimator_.predict(X_test)
    )

    predicted_changes = (
        search.best_estimator_
        .predict_residual(X_test)
    )

    predictions_by_model = {
        "Persistence": persistence_predictions,
        "Original XGBoost": original_predictions,
        "Optimized XGBoost": optimized_predictions,
        "Residual XGBoost": residual_predictions,
    }

    all_test_metrics = {
        name: evaluate_predictions(
            y_test,
            predictions,
            threshold,
        )
        for name, predictions
        in predictions_by_model.items()
    }

    comparison = pd.DataFrame([
        {
            "Model": name,
            **metrics,
        }
        for name, metrics
        in all_test_metrics.items()
    ])

    comparison.to_csv(
        FINAL_MODEL_COMPARISON_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FINAL TEST SET COMPARISON")
    print("=" * 80)

    print(
        comparison.round(4)
        .to_string(index=False)
    )

    # 7. Save prediction records

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
        "residual_prediction_wh": (
            residual_predictions
        ),
        "predicted_change_wh": (
            predicted_changes
        ),
        "actual_change_wh": (
            y_test.to_numpy()
            - persistence_predictions
        ),
    })

    predictions.to_csv(
        RESIDUAL_PREDICTIONS_PATH,
        index=False,
    )

    # 8. Compare energy policies

    energy_results = [
        evaluate_energy_policy(
            y_test,
            np.zeros(len(y_test)),
            threshold,
            "Always Normal",
        )
    ]

    for name, model_predictions in (
        predictions_by_model.items()
    ):

        energy_results.append(
            evaluate_energy_policy(
                y_test,
                model_predictions,
                threshold,
                f"Adaptive {name}",
            )
        )

    energy_comparison = pd.DataFrame(
        energy_results
    )

    energy_comparison.to_csv(
        ENERGY_POLICY_COMPARISON_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("ENERGY POLICY COMPARISON")
    print("=" * 80)

    print(
        energy_comparison.round(4)
        .to_string(index=False)
    )

    # 9. Save final metrics

    metrics = {
        "cross_validation": all_cv_metrics,
        "test_metrics": all_test_metrics,
        "model_selection": selection,
        "best_parameters": search.best_params_,
        "tuning_time_seconds": tuning_time,
        "refit_time_seconds": float(
            search.refit_time_
        ),
        "eco_threshold_wh": float(threshold),
        "energy_policy_comparison": (
            energy_comparison.to_dict(
                orient="records"
            )
        ),
    }

    with open(
        RESIDUAL_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # 10. Final output

    print("\nSaved artifacts:")

    for path in [
        RESIDUAL_MODEL_PATH,
        RESIDUAL_METADATA_PATH,
        RESIDUAL_TUNING_RESULTS_PATH,
        RESIDUAL_BEST_PARAMETERS_PATH,
        RESIDUAL_PREDICTIONS_PATH,
        RESIDUAL_METRICS_PATH,
        FINAL_MODEL_COMPARISON_PATH,
        FINAL_CV_COMPARISON_PATH,
        MODEL_SELECTION_PATH,
        ENERGY_POLICY_COMPARISON_PATH,
    ]:

        print(f"  {path}")

    print("\nResidual Optimization: COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
