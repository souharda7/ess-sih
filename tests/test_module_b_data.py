from __future__ import annotations

import pandas as pd

from ess_module_b.config import default_config
from ess_module_b.data import prepare_series


def _wide_row() -> dict:
    return {
        "component_id": "C1",
        "lot_id": "L1",
        "part_number": "PN_LOGIC_A",
        "parameter_name": "leakage_current",
        "unit": "uA",
        "test_condition_id": "PN_LOGIC_A_125C_NOMINAL",
        "temperature_C": 125.0,
        "bias_voltage": 3.3,
        "test_mode": "static_bias",
        "Value_0h": 10.0,
        "Value_24h": 11.0,
        "actual_value_168h": 14.0,
    }


def test_wide_input_and_problem_statement_aliases_are_supported():
    prepared = prepare_series(
        pd.DataFrame([_wide_row()]),
        default_config(),
        include_target=True,
        require_single_lot=False,
    )
    row = prepared.series.iloc[0]
    assert row["parameter"] == "leakage_current"
    assert row["value_0h"] == 10.0
    assert row["value_24h"] == 11.0
    assert row["actual_value_168h"] == 14.0
    assert row["data_quality_status"] == "VALID"


def test_96_hour_rows_are_not_used_by_module_b(test_lot):
    prepared = prepare_series(
        test_lot,
        default_config(),
        include_target=False,
        require_single_lot=True,
    )
    assert prepared.ignored_measurement_count == len(test_lot) // 2
    assert all(item in prepared.series.columns for item in ["value_0h", "value_24h"])


def test_hidden_wide_target_is_removed_during_forecasting():
    prepared = prepare_series(
        pd.DataFrame([_wide_row()]),
        default_config(),
        include_target=False,
        require_single_lot=True,
    )
    assert "actual_value_168h" not in prepared.series.columns


def test_checkpoint_completeness_is_scoped_by_lot_when_component_ids_repeat():
    rows = []
    for lot_id, checkpoints in [("L1", [0, 24]), ("L2", [0])]:
        for time_h in checkpoints:
            row = _wide_row()
            row.pop("Value_0h")
            row.pop("Value_24h")
            row.pop("actual_value_168h")
            row["lot_id"] = lot_id
            row["time_h"] = time_h
            row["value"] = 10.0
            rows.append(row)
    prepared = prepare_series(
        pd.DataFrame(rows),
        default_config(),
        include_target=False,
        require_single_lot=False,
    )
    status = prepared.series.set_index("lot_id")["data_quality_status"].to_dict()
    assert status == {"L1": "VALID", "L2": "INVALID"}
