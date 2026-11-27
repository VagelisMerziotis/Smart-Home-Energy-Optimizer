"""Shared pytest fixtures.

Dataset and preprocessing fixtures are session-scoped because loading the
CSV and building lag features is relatively expensive. PolicyEngine is
function-scoped so tests cannot leak mutated controller state.
"""

import pytest

from data_loader import load_dataset
from preprocessing import prepare_data
from policy_engine import PolicyEngine


@pytest.fixture(scope="session")
def raw_dataset():
    """Validated UCI series used by data-loader and lag-alignment tests."""
    return load_dataset()


@pytest.fixture(scope="session")
def prepared_data():
    """Tuple of (train, test, eco_threshold) from the production pipeline."""
    return prepare_data()


@pytest.fixture(scope="session")
def train_data(prepared_data):
    """Featured training table (target timestamps before the cutoff)."""
    return prepared_data[0]


@pytest.fixture(scope="session")
def test_data(prepared_data):
    """Featured test table (3,947 ten-minute intervals)."""
    return prepared_data[1]


@pytest.fixture(scope="session")
def eco_threshold(prepared_data):
    """Training-only 75th percentile of Appliances, expected to be 100 Wh."""
    return prepared_data[2]


@pytest.fixture
def policy_engine(eco_threshold):
    """Fresh PolicyEngine using the pipeline Eco threshold."""
    return PolicyEngine(eco_threshold=eco_threshold)