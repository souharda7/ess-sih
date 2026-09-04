from __future__ import annotations

import pandas as pd
import pytest

from ess_module_a.validation import DataValidationError, validate_measurements


def _measurement(component: str, time_h: float = 0, unit: str = "uA") -> dict:
    return {
        "component_id": component,
        "lot_id": "L1",
        "part_number": "PN_LOGIC_A",
        "parameter": "leakage_current",
        "time_h": time_h,
        "value": 10.0,
        "unit": unit,
        "test_condition_id": "PN_LOGIC_A_125C_NOMINAL",
        "temperature_c": 125.0,
        "voltage_v": 3.3,
        "test_mode": "static_bias",
    }


def test_missing_columns_rejected(config):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_measurements(pd.DataFrame([{"component_id": "C1"}]), config)


def test_duplicate_and_bad_unit_are_series_issues(config):
    rows = [_measurement("C1"), _measurement("C1")]
    rows.append(_measurement("C2", unit="furlong"))
    result = validate_measurements(pd.DataFrame(rows), config, as_of_h=0)
    codes = {issue.code for issue in result.issues}
    assert "DUPLICATE_MEASUREMENT" in codes
    assert "UNIT_OR_TRANSFORM_ERROR" in codes


def test_future_measurements_are_ignored_not_series_failures(config):
    rows = [_measurement("C1", 0), _measurement("C1", 24), _measurement("C1", 168)]
    result = validate_measurements(pd.DataFrame(rows), config, as_of_h=24)
    assert "FUTURE_MEASUREMENT_IGNORED" in {issue.code for issue in result.issues}
    key = ("C1", "leakage_current", "PN_LOGIC_A_125C_NOMINAL")
    assert "FUTURE_MEASUREMENT_IGNORED" not in result.series_issues.get(key, [])
