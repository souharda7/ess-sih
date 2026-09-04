from __future__ import annotations

import pytest

from ess_module_b.models import DriftDecision


def _prediction(report, component_id, parameter):
    return next(
        item
        for item in report["prediction_results"]
        if item["component_id"] == component_id and item["parameter"] == parameter
    )


def test_forecast_uses_only_0h_and_24h(trained_module_b_engine, test_lot):
    early = test_lot.loc[test_lot["time_h"] <= 24].copy()
    early_report = trained_module_b_engine.forecast_lot(early)
    full_report = trained_module_b_engine.forecast_lot(test_lot)
    early_values = {
        (item["component_id"], item["parameter"]): item["predicted_value_168h"]
        for item in early_report["prediction_results"]
    }
    full_values = {
        (item["component_id"], item["parameter"]): item["predicted_value_168h"]
        for item in full_report["prediction_results"]
    }
    assert full_values == early_values
    assert full_report["ignored_measurement_count"] == len(test_lot) // 2


def test_required_baselines_and_explanation_are_exact(trained_module_b_engine, test_lot):
    report = trained_module_b_engine.forecast_lot(test_lot)
    component = str(test_lot["component_id"].iloc[0])
    result = _prediction(report, component, "leakage_current")
    assert result["persistence_prediction_168h"] == result["value_24h"]
    assert result["linear_extrapolation_prediction_168h"] == pytest.approx(
        7 * result["value_24h"] - 6 * result["value_0h"]
    )
    contribution = result["explanation"]["contributions"]
    reconstructed = (
        contribution["intercept"]
        + sum(item["contribution"] for item in contribution["terms"])
        + contribution["postprocessing_adjustment"]
    )
    assert reconstructed == pytest.approx(result["predicted_value_168h"])
    assert result["explanation"]["calibration"]["cross_validated_mae"] >= 0


def test_delta_limit_can_be_the_binding_safety_slope(trained_module_b_engine, test_lot):
    lot = test_lot.copy()
    component = str(lot["component_id"].iloc[0])
    mask = (lot["component_id"] == component) & (lot["parameter"] == "leakage_current")
    lot.loc[mask, "delta_limit"] = 0.0001
    report = trained_module_b_engine.forecast_lot(lot)
    result = _prediction(report, component, "leakage_current")
    assert result["safety_slope_per_h"] == pytest.approx(0.0001 / 168.0)
    assert any(
        source["source"] == "configured_delta_limit" and source["binding"]
        for source in result["safety_slope_sources"]
    )


def test_conflicting_engineering_limits_require_retest(trained_module_b_engine, test_lot):
    lot = test_lot.copy()
    component = str(lot["component_id"].iloc[0])
    base_mask = (lot["component_id"] == component) & (
        lot["parameter"] == "leakage_current"
    )
    lot.loc[base_mask & (lot["time_h"] == 0), "delta_limit"] = 1.0
    lot.loc[base_mask & (lot["time_h"] == 24), "delta_limit"] = 2.0
    result = _prediction(
        trained_module_b_engine.forecast_lot(lot), component, "leakage_current"
    )
    assert result["decision"] == DriftDecision.RETEST_REQUIRED.value
    assert "CONFLICTING_DELTA_LIMIT" in result["reason_codes"]


def test_early_static_failure_has_precedence(trained_module_b_engine, test_lot):
    lot = test_lot.copy()
    component = str(lot["component_id"].iloc[0])
    mask = (
        (lot["component_id"] == component)
        & (lot["parameter"] == "leakage_current")
        & (lot["time_h"] == 24)
    )
    lot.loc[mask, "value"] = 60.0
    report = trained_module_b_engine.forecast_lot(lot)
    result = _prediction(report, component, "leakage_current")
    assert result["decision"] == DriftDecision.STATIC_FAIL.value
    assert "STATIC_LIMIT_HIGH_24H" in result["reason_codes"]


def test_rapid_early_drift_is_rejected_early(trained_module_b_engine, test_lot):
    lot = test_lot.copy()
    component = str(lot["component_id"].iloc[0])
    mask = (
        (lot["component_id"] == component)
        & (lot["parameter"] == "leakage_current")
        & (lot["time_h"] == 24)
    )
    lot.loc[mask, "value"] = lot.loc[mask, "value"] + 8.0
    report = trained_module_b_engine.forecast_lot(lot)
    summary = next(
        item for item in report["component_results"] if item["component_id"] == component
    )
    assert summary["flagged_for_early_rejection"] is True
