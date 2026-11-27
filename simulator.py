"""Replay the held-out test set in virtual time through model and policy.

Each step uses only features known at decision time, chooses Normal/Eco from
the forecast, then reveals the historical actual consumption to score the
policy. Simulated Eco consumption is not fed back as a later model input.
"""

import json
import time

import numpy as np
import pandas as pd

from config import (
    DATE_COLUMN,
    RESULTS_DIR,
    SIMULATION_LOG_PATH,
    SIMULATION_METRICS_PATH,
)

from preprocessing import (
    prepare_data,
    FEATURE_COLUMNS,
    TARGET_NAME,
    TARGET_TIME_COLUMN,
)

from model_manager import ModelManager

from policy_engine import (
    PolicyEngine,
    EnergyMode,
)


class EnergySimulator:
    """Stepwise test-set replay used by the CLI, tests, and Streamlit live page."""

    def __init__(self, model_name=None):

        train, test, threshold = prepare_data()

        self.model_manager = ModelManager(
            model_name=model_name
        )

        self.model_name = (
            self.model_manager.model_name
        )

        self.metadata = (
            self.model_manager.metadata
        )

        self.validate_model_data(
            train,
            test,
            threshold,
        )

        self.test = test.reset_index(
            drop=True
        )

        self.controller = PolicyEngine(
            eco_threshold=threshold
        )

        self.reset()

    # --------------------------------------------------
    # Data and model consistency
    # --------------------------------------------------

    def validate_model_data(
        self,
        train,
        test,
        threshold,
    ):
        """Ensure loaded metadata still matches the current preprocessing split."""


        if not np.isclose(
            threshold,
            self.metadata["eco_threshold_wh"],
        ):

            raise ValueError(
                "Eco threshold mismatch."
            )

        if (
            self.metadata["training_samples"]
            != len(train)
        ):

            raise ValueError(
                "Training sample count mismatch."
            )

        if (
            self.metadata["test_samples"]
            != len(test)
        ):

            raise ValueError(
                "Test sample count mismatch."
            )

        expected_dates = {
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
        }

        for key, expected in expected_dates.items():

            if self.metadata[key] != expected:

                raise ValueError(
                    f"Timestamp mismatch: {key}"
                )

    # --------------------------------------------------
    # Simulation state
    # --------------------------------------------------

    @property
    def total_steps(self):
        """Number of 10-minute test intervals available to replay."""


        return len(self.test)

    @property
    def finished(self):
        """True once every test interval has been processed."""


        return (
            self.current_step
            >= self.total_steps
        )

    @property
    def progress(self):
        """Fraction of the test set already replayed, in [0, 1]."""


        return (
            self.current_step / self.total_steps
            if self.total_steps > 0
            else 0.0
        )

    def reset(self):
        """Clear history and return to the first test interval."""


        self.current_step = 0

        self.history = []

    # --------------------------------------------------
    # Single simulation step
    # --------------------------------------------------

    def step(self):
        """Advance one interval: predict, decide, then score against actual."""


        if self.finished:

            return None

        start_time = time.perf_counter()

        index = self.current_step

        # 1. Read information available at time t

        features = self.test.iloc[[index]][
            FEATURE_COLUMNS
        ]

        decision_time = self.test.at[
            index,
            DATE_COLUMN,
        ]

        target_time = self.test.at[
            index,
            TARGET_TIME_COLUMN,
        ]

        # 2. Predict consumption at t + 10 minutes

        prediction = (
            self.model_manager.predict_one(
                features
            )
        )

        baseline_prediction = float(
            features[
                "appliances_lag_0"
            ].iloc[0]
        )

        # 3. Decide energy mode before revealing actual

        mode = self.controller.decide(
            prediction
        )

        # 4. Reveal historical actual consumption

        actual = float(
            self.test.at[
                index,
                TARGET_NAME,
            ]
        )

        # 5. Apply hypothetical energy policy

        savings = (
            self.controller.calculate_savings(
                actual,
                mode,
            )
        )

        simulated = actual - savings

        # 6. Create simulation record

        record = {
            "step": index + 1,
            "decision_time": decision_time,
            "target_time": target_time,
            "model_name": self.model_name,
            "prediction_type": (
                self.model_manager.prediction_type
            ),
            "actual_wh": actual,
            "xgboost_prediction_wh": prediction,
            "baseline_prediction_wh": (
                baseline_prediction
            ),
            "mode": mode.value,
            "eco_threshold_wh": (
                self.controller.eco_threshold
            ),
            "always_normal_wh": actual,
            "adaptive_consumption_wh": simulated,
            "hypothetical_savings_wh": savings,
        }

        record["processing_time_ms"] = (
            time.perf_counter() - start_time
        ) * 1000

        self.history.append(record)

        self.current_step += 1

        return record

    # --------------------------------------------------
    # Full simulation
    # --------------------------------------------------

    def run_all(self):
        """Replay the remaining test intervals and return the history table."""


        while not self.finished:

            self.step()

        return self.get_history()

    def get_history(self):
        """Return one row per completed step, or an empty DataFrame."""


        return pd.DataFrame(
            self.history
        )

    # --------------------------------------------------
    # Simulation metrics
    # --------------------------------------------------

    def get_summary(self):
        """Aggregate mode counts, hypothetical energy, and core latency."""


        data = self.get_history()

        summary = {
            "model_name": self.model_name,
            "prediction_type": (
                self.model_manager.prediction_type
            ),
            "eco_threshold_wh": (
                self.controller.eco_threshold
            ),
            "intervals": 0,
            "normal_intervals": 0,
            "eco_intervals": 0,
            "eco_percent": 0.0,
            "always_normal_wh": 0.0,
            "adaptive_consumption_wh": 0.0,
            "hypothetical_savings_wh": 0.0,
            "hypothetical_savings_percent": 0.0,
            "mean_processing_time_ms": 0.0,
            "max_processing_time_ms": 0.0,
        }

        if data.empty:

            return summary

        total_normal = float(
            data["always_normal_wh"].sum()
        )

        total_adaptive = float(
            data["adaptive_consumption_wh"].sum()
        )

        total_savings = float(
            data["hypothetical_savings_wh"].sum()
        )

        eco_intervals = int(
            (
                data["mode"]
                == EnergyMode.ECO.value
            ).sum()
        )

        total_intervals = len(data)

        savings_percent = (
            total_savings / total_normal * 100
            if total_normal > 0
            else 0.0
        )

        summary.update({
            "intervals": total_intervals,
            "normal_intervals": (
                total_intervals - eco_intervals
            ),
            "eco_intervals": eco_intervals,
            "eco_percent": (
                eco_intervals
                / total_intervals
                * 100
            ),
            "always_normal_wh": total_normal,
            "adaptive_consumption_wh": (
                total_adaptive
            ),
            "hypothetical_savings_wh": (
                total_savings
            ),
            "hypothetical_savings_percent": (
                savings_percent
            ),
            "mean_processing_time_ms": float(
                data["processing_time_ms"].mean()
            ),
            "max_processing_time_ms": float(
                data["processing_time_ms"].max()
            ),
        })

        return summary

    # --------------------------------------------------
    # Save simulation results
    # --------------------------------------------------

    def save_results(self):
        """Write the interval log and summary JSON used by evaluation scripts."""


        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.get_history().to_csv(
            SIMULATION_LOG_PATH,
            index=False,
        )

        with open(
            SIMULATION_METRICS_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.get_summary(),
                file,
                indent=4,
            )


# --------------------------------------------------
# Console output
# --------------------------------------------------

def print_summary(summary):
    """Print energy and latency totals for a completed CLI run."""


    print("\n" + "=" * 65)
    print("SIMULATION RESULTS")
    print("=" * 65)

    print(
        f"Selected model: "
        f"{summary['model_name']}"
    )

    print(
        f"Prediction type: "
        f"{summary['prediction_type']}"
    )

    print(
        f"Total intervals: "
        f"{summary['intervals']:,}"
    )

    print(
        f"Normal intervals: "
        f"{summary['normal_intervals']:,}"
    )

    print(
        f"Eco intervals: "
        f"{summary['eco_intervals']:,}"
    )

    print(
        f"Eco activation: "
        f"{summary['eco_percent']:.2f}%"
    )

    print(
        f"\nAlways Normal: "
        f"{summary['always_normal_wh']:,.2f} Wh"
    )

    print(
        f"Adaptive Normal/Eco: "
        f"{summary['adaptive_consumption_wh']:,.2f} Wh"
    )

    print(
        f"Hypothetical savings: "
        f"{summary['hypothetical_savings_wh']:,.2f} Wh"
    )

    print(
        f"Hypothetical savings percentage: "
        f"{summary['hypothetical_savings_percent']:.2f}%"
    )

    print(
        f"\nMean processing time: "
        f"{summary['mean_processing_time_ms']:.3f} ms"
    )

    print(
        f"Max processing time: "
        f"{summary['max_processing_time_ms']:.3f} ms"
    )


def main():
    """Run the full test replay for the selected model and save artifacts."""


    print("=" * 65)
    print("SMART HOME ENERGY OPTIMIZER")
    print("Final Model Simulation")
    print("=" * 65)

    simulator = EnergySimulator()

    print(
        f"Selected model: "
        f"{simulator.model_name}"
    )

    print(
        f"Total steps: "
        f"{simulator.total_steps:,}"
    )

    print(
        f"Eco threshold: "
        f"{simulator.controller.eco_threshold:.2f} Wh"
    )

    start_time = time.perf_counter()

    simulator.run_all()

    elapsed = (
        time.perf_counter() - start_time
    )

    simulator.save_results()

    print_summary(
        simulator.get_summary()
    )

    print(
        f"\nTotal execution time: "
        f"{elapsed:.3f} sec"
    )

    print("\nSaved artifacts:")

    print(f"  {SIMULATION_LOG_PATH}")

    print(f"  {SIMULATION_METRICS_PATH}")

    print("\nSimulation: COMPLETED")


if __name__ == "__main__":
    main()
