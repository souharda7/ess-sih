from __future__ import annotations

import pandas as pd
import pytest

from ess_module_a.config import default_config
from ess_module_a.engine import ModuleAEngine
from ess_module_a.synthetic import generate_synthetic_data


@pytest.fixture(scope="session")
def config():
    return default_config()


@pytest.fixture(scope="session")
def trained_engine(config):
    training = generate_synthetic_data(n_lots=8, components_per_lot=40, seed=170)
    engine = ModuleAEngine(config)
    engine.fit(training)
    return engine


@pytest.fixture()
def test_lot() -> pd.DataFrame:
    frame = generate_synthetic_data(n_lots=3, components_per_lot=40, seed=999)
    lot = frame.loc[frame["lot_id"] == "LOT_001"].copy()
    lot["lot_id"] = "TEST_LOT_001"
    lot["component_id"] = lot["component_id"].str.replace("LOT_001", "TEST_LOT_001")
    return lot
