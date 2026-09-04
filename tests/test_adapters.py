from __future__ import annotations

import pandas as pd
import pytest

from ess_module_a.adapters import (
    NASALeakageSnapshot,
    adapt_nasa_igbt_leakage,
    adapt_wide_checkpoints,
)


def test_nasa_leakage_adapter_selects_nearest_voltage(tmp_path):
    path = tmp_path / "LeakageIV.csv"
    pd.DataFrame([[1.0, 1e-8], [10.0, 2e-7], [20.0, 5e-7]]).to_csv(
        path, index=False, header=False
    )
    result = adapt_nasa_igbt_leakage(
        [NASALeakageSnapshot("DEVICE_1", 0, path)], target_voltage_v=11.0
    )
    assert result.iloc[0]["value"] == pytest.approx(2e-7)
    assert result.iloc[0]["unit"] == "A"
    assert result.iloc[0]["voltage_v"] == 10.0


def test_wide_adapter_creates_one_row_per_checkpoint():
    wide = pd.DataFrame(
        [{"component_id": "C1", "lot_id": "L1", "Value_0h": 10, "Value_24h": 11}]
    )
    result = adapt_wide_checkpoints(
        wide,
        id_columns=["component_id", "lot_id"],
        checkpoint_columns={"Value_0h": 0, "Value_24h": 24},
        parameter="leakage_current",
        unit="uA",
    )
    assert result["time_h"].tolist() == [0.0, 24.0]
    assert result["value"].tolist() == [10, 11]
