"""Select Normal/Eco from the forecast, then apply a hypothetical load cut.

The controller never sees the future actual consumption when choosing a
mode. Savings are an accounting identity used for simulation:

    savings = actual_wh * CONTROLLABLE_LOAD_RATIO * ECO_REDUCTION_RATIO

when Eco is selected, otherwise zero. This is not a measured device
response.
"""

import math

from enum import Enum

from config import (
    CONTROLLABLE_LOAD_RATIO,
    ECO_REDUCTION_RATIO,
)


class EnergyMode(str, Enum):
    """Operating modes used by the simulator and dashboard."""

    NORMAL = "Normal"
    ECO = "Eco"


class PolicyEngine:
    """Threshold policy: Eco only if the forecast exceeds `eco_threshold`."""

    def __init__(self, eco_threshold):
        self.eco_threshold = float(eco_threshold)

        if (
            not math.isfinite(self.eco_threshold)
            or self.eco_threshold < 0
        ):
            raise ValueError("Invalid Eco threshold.")

    def decide(self, predicted_wh):
        """Return Eco when predicted consumption is strictly above threshold.

        Equality stays in Normal so the 75th-percentile cutoff is exclusive
        on the high side.
        """

        predicted_wh = float(predicted_wh)

        if (
            not math.isfinite(predicted_wh)
            or predicted_wh < 0
        ):
            raise ValueError("Invalid prediction.")

        if predicted_wh > self.eco_threshold:
            return EnergyMode.ECO

        return EnergyMode.NORMAL

    def calculate_savings(self, actual_wh, mode):
        """Return hypothetical Wh saved under the selected mode."""

        actual_wh = float(actual_wh)

        if (
            not math.isfinite(actual_wh)
            or actual_wh < 0
        ):
            raise ValueError("Invalid consumption.")

        mode = EnergyMode(mode)

        if mode == EnergyMode.ECO:
            return (
                actual_wh
                * CONTROLLABLE_LOAD_RATIO
                * ECO_REDUCTION_RATIO
            )

        return 0.0

    def simulate_consumption(self, actual_wh, mode):
        """Return recorded consumption minus hypothetical Eco savings."""

        savings = self.calculate_savings(
            actual_wh,
            mode,
        )

        return float(actual_wh) - savings


def main():
    """Print a few hand-crafted Normal/Eco scenarios for a sanity check."""

    controller = PolicyEngine(eco_threshold=100)

    scenarios = [
        (80, 90),
        (100, 150),
        (130, 200),
        (250, 300),
    ]

    print("=" * 65)
    print("POLICY ENGINE — FUNCTIONAL SCENARIOS")
    print("=" * 65)

    for prediction, actual in scenarios:

        mode = controller.decide(prediction)

        savings = controller.calculate_savings(
            actual,
            mode,
        )

        simulated = controller.simulate_consumption(
            actual,
            mode,
        )

        print(f"\nPrediction: {prediction:.2f} Wh")
        print(f"Selected mode: {mode.value}")
        print(f"Actual consumption: {actual:.2f} Wh")
        print(f"Hypothetical savings: {savings:.2f} Wh")
        print(f"Simulated consumption: {simulated:.2f} Wh")

    print("\nPolicy scenarios: COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
