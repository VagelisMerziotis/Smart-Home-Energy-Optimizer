"""Shared project paths, dataset fields, and modelling constants.

All training, evaluation, simulation, and dashboard scripts import from this
module so that split ratios, lag features, Eco policy assumptions, and output
locations stay consistent. Changing a value here affects the whole pipeline.
"""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "energydata_complete.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Dataset configuration
DATE_COLUMN = "date"
TARGET_COLUMN = "Appliances"
SAMPLING_INTERVAL_MINUTES = 10

# Prediction configuration
# Forecast one sampling step ahead (t + 10 minutes).
FORECAST_HORIZON_MINUTES = 10
FORECAST_STEPS = 1
TRAIN_RATIO = 0.80
RANDOM_STATE = 42

# Indoor/outdoor temperature and humidity used as contemporaneous inputs.
ENVIRONMENTAL_FEATURES = [
    "T1",
    "T2",
    "T_out",
    "RH_1",
    "RH_2",
]

# Consumption lags in sampling steps: t, t-10 min, t-20 min.
LAG_STEPS = [0, 1, 2]

# Eco policy configuration
# Threshold is the given percentile of training-period Appliances (Wh).
ECO_PERCENTILE = 0.75
# Fraction of recorded load assumed to be controllable in Eco mode.
CONTROLLABLE_LOAD_RATIO = 0.30
# Fraction of that controllable load hypothetically reduced in Eco mode.
ECO_REDUCTION_RATIO = 0.20

# Dashboard playback: wall-clock seconds between virtual 10-minute steps.
SIMULATION_STEP_SECONDS = 1.0

# Initial (untuned) XGBoost hyperparameters used by train_model.py.
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Original model and evaluation output paths
MODEL_PATH = MODELS_DIR / "xgboost_model.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# Energy simulation output paths
SIMULATION_LOG_PATH = RESULTS_DIR / "simulation_log.csv"
SIMULATION_METRICS_PATH = RESULTS_DIR / "simulation_metrics.json"

# Hyperparameter optimization (direct XGBoost)
TUNING_ITERATIONS = 100
TUNING_CV_SPLITS = 5
TUNING_N_JOBS = -1

OPTIMIZED_MODEL_PATH = MODELS_DIR / "xgboost_optimized.json"
OPTIMIZED_METADATA_PATH = MODELS_DIR / "optimized_metadata.json"
TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results.csv"
BEST_PARAMETERS_PATH = RESULTS_DIR / "best_parameters.json"
OPTIMIZED_PREDICTIONS_PATH = RESULTS_DIR / "optimized_predictions.csv"
OPTIMIZED_METRICS_PATH = RESULTS_DIR / "optimized_metrics.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

# Residual XGBoost optimization
RESIDUAL_MODEL_PATH = MODELS_DIR / "xgboost_residual.json"
RESIDUAL_METADATA_PATH = MODELS_DIR / "residual_metadata.json"
RESIDUAL_TUNING_RESULTS_PATH = RESULTS_DIR / "residual_tuning_results.csv"
RESIDUAL_BEST_PARAMETERS_PATH = RESULTS_DIR / "residual_best_parameters.json"
RESIDUAL_PREDICTIONS_PATH = RESULTS_DIR / "residual_predictions.csv"
RESIDUAL_METRICS_PATH = RESULTS_DIR / "residual_metrics.json"
FINAL_MODEL_COMPARISON_PATH = RESULTS_DIR / "final_model_comparison.csv"
FINAL_CV_COMPARISON_PATH = RESULTS_DIR / "final_cv_comparison.csv"
MODEL_SELECTION_PATH = RESULTS_DIR / "model_selection.json"
ENERGY_POLICY_COMPARISON_PATH = RESULTS_DIR / "energy_policy_comparison.csv"

# Final evaluation and visualization
FINAL_FIGURES_DIR = RESULTS_DIR / "final_figures"
FINAL_RANGE_METRICS_PATH = RESULTS_DIR / "final_consumption_range_metrics.csv"
FINAL_FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "final_feature_importance.csv"
PERFORMANCE_METRICS_PATH = RESULTS_DIR / "performance_metrics.json"
