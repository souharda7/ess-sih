from __future__ import annotations

import pandas as pd

from ess_module_a.models import QAStatus
from ess_module_a.synthetic import generate_synthetic_data


def _through(frame: pd.DataFrame, time_h: float) -> pd.DataFrame:
    return frame.loc[frame["time_h"] <= time_h].copy()


def test_high_in_spec_value_is_dynamically_quarantined(trained_engine, test_lot):
    lot = _through(test_lot, 24)
    component = lot["component_id"].iloc[0]
    mask = (
        (lot["component_id"] == component)
        & (lot["parameter"] == "leakage_current")
        & (lot["time_h"] == 24)
    )
    lot.loc[mask, "value"] = 45.0
    report = trained_engine.score_lot(lot, as_of_h=24)
    result = next(
        item
        for item in report["parameter_results"]
        if item["component_id"] == component
        and item["parameter"] == "leakage_current"
        and item["time_h"] == 24
    )
    assert result["normalized_value"] == 45.0
    assert result["static_status"] == "PASS"
    assert result["status"] == QAStatus.QUARANTINE.value
    assert result["robust_z_lot"] > 5
    assert "LOT_ROBUST_Z_EXTREME" in result["reason_codes"]


def test_static_failure_has_highest_precedence(trained_engine, test_lot):
    lot = _through(test_lot, 24)
    component = lot["component_id"].iloc[0]
    mask = (
        (lot["component_id"] == component)
        & (lot["parameter"] == "leakage_current")
        & (lot["time_h"] == 24)
    )
    lot.loc[mask, "value"] = 60.0
    report = trained_engine.score_lot(lot, as_of_h=24)
    summary = next(item for item in report["component_results"] if item["component_id"] == component)
    assert summary["status"] == QAStatus.STATIC_FAIL.value


def test_missing_checkpoint_requires_retest(trained_engine, test_lot):
    lot = _through(test_lot, 24)
    component = lot["component_id"].iloc[0]
    mask = (
        (lot["component_id"] == component)
        & (lot["parameter"] == "leakage_current")
        & (lot["time_h"] == 24)
    )
    report = trained_engine.score_lot(lot.loc[~mask], as_of_h=24)
    summary = next(item for item in report["component_results"] if item["component_id"] == component)
    assert summary["status"] == QAStatus.RETEST_REQUIRED.value
    assert "MISSING_CHECKPOINT" in summary["reason_codes"]


def test_lower_is_dangerous_for_output_high_voltage(trained_engine, test_lot):
    lot = _through(test_lot, 24)
    component = lot["component_id"].iloc[0]
    mask = (
        (lot["component_id"] == component)
        & (lot["parameter"] == "output_high_voltage")
        & (lot["time_h"] == 24)
    )
    lot.loc[mask, "value"] = 2.45
    report = trained_engine.score_lot(lot, as_of_h=24)
    result = next(
        item
        for item in report["parameter_results"]
        if item["component_id"] == component
        and item["parameter"] == "output_high_voltage"
        and item["time_h"] == 24
    )
    assert result["static_status"] == "PASS"
    assert result["status"] == QAStatus.QUARANTINE.value


def test_full_data_can_be_scored_as_of_24_without_leakage(trained_engine, test_lot):
    report = trained_engine.score_lot(test_lot, as_of_h=24)
    assert all(item["time_h"] <= 24 for item in report["parameter_results"])
    future_codes = {issue["code"] for issue in report["validation_issues"]}
    assert "FUTURE_MEASUREMENT_IGNORED" in future_codes
    assert not all(item["status"] == QAStatus.RETEST_REQUIRED.value for item in report["component_results"])


def test_whole_lot_shift_creates_lot_alert(trained_engine):
    frame = generate_synthetic_data(n_lots=3, components_per_lot=40, seed=777)
    lot = frame.loc[frame["lot_id"] == "LOT_003"].copy()
    lot["lot_id"] = "SHIFTED_LOT"
    lot["component_id"] = lot["component_id"].str.replace("LOT_003", "SHIFTED_LOT")
    report = trained_engine.score_lot(_through(lot, 24), as_of_h=24)
    assert any(alert["type"] == "WHOLE_LOT_SHIFT" for alert in report["lot_alerts"])
