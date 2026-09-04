from __future__ import annotations

import pytest

from ess_ensemble import EnsembleDecision, EnsembleEngine, combine_reports
from ess_ensemble.evaluation import evaluate_partition


def _module_a_component(
    component_id: str = "C001",
    *,
    status: str = "NORMAL",
    risk: float = 0.2,
) -> dict:
    return {
        "component_id": component_id,
        "status": status,
        "risk_score": risk,
        "highest_risk_parameter": "leakage_current",
        "reason_codes": [f"A_{status}"],
    }


def _module_b_component(
    component_id: str = "C001",
    *,
    decision: str = "CONTINUE_SCREENING",
    risk: float = 0.3,
) -> dict:
    return {
        "component_id": component_id,
        "decision": decision,
        "risk_score": risk,
        "highest_risk_parameter": "iddq",
        "predicted_value_168h": 42.0,
        "reason_codes": [f"B_{decision}"],
    }


def _reports(a_component: dict, b_component: dict) -> tuple[dict, dict]:
    return (
        {
            "lot_id": "LOT_TEST",
            "as_of_h": 24.0,
            "component_results": [a_component],
            "validation_issues": [],
        },
        {
            "lot_id": "LOT_TEST",
            "input_as_of_h": 24.0,
            "target_h": 168.0,
            "component_results": [b_component],
            "validation_issues": [],
        },
    )


@pytest.mark.parametrize(
    ("module_a_status", "module_b_decision", "expected"),
    [
        ("NORMAL", "CONTINUE_SCREENING", EnsembleDecision.CONTINUE_SCREENING),
        ("MONITOR", "CONTINUE_SCREENING", EnsembleDecision.MONITOR),
        ("RETEST_REQUIRED", "CONTINUE_SCREENING", EnsembleDecision.RETEST_REQUIRED),
        ("NORMAL", "RETEST_REQUIRED", EnsembleDecision.RETEST_REQUIRED),
        ("QUARANTINE", "CONTINUE_SCREENING", EnsembleDecision.REJECT_EARLY),
        ("NORMAL", "EARLY_REJECT", EnsembleDecision.REJECT_EARLY),
        ("STATIC_FAIL", "RETEST_REQUIRED", EnsembleDecision.REJECT_EARLY),
        ("RETEST_REQUIRED", "STATIC_FAIL", EnsembleDecision.REJECT_EARLY),
    ],
)
def test_safety_first_decision_matrix(
    module_a_status: str,
    module_b_decision: str,
    expected: EnsembleDecision,
):
    report = combine_reports(
        *_reports(
            _module_a_component(status=module_a_status),
            _module_b_component(decision=module_b_decision),
        )
    )
    assert report["component_results"][0]["final_decision"] == expected.value


def test_fused_risk_does_not_dilute_either_module_and_preserves_evidence():
    report = combine_reports(
        *_reports(
            _module_a_component(status="QUARANTINE", risk=0.6),
            _module_b_component(decision="EARLY_REJECT", risk=0.7),
        )
    )
    result = report["component_results"][0]
    assert result["ensemble_risk_score"] == pytest.approx(0.88)
    assert result["ensemble_risk_score"] >= result["module_a_risk_score"]
    assert result["ensemble_risk_score"] >= result["module_b_risk_score"]
    assert result["evidence_pattern"] == "BOTH_MODULES_REJECT"
    assert "CROSS_MODULE_REJECT_CONSENSUS" in result["reason_codes"]
    assert result["module_a_reason_codes"] == ["A_QUARANTINE"]
    assert result["module_b_reason_codes"] == ["B_EARLY_REJECT"]


def test_missing_module_result_requires_retest_instead_of_silent_clearance():
    module_a, module_b = _reports(
        _module_a_component("C001"),
        _module_b_component("C002"),
    )
    report = EnsembleEngine().combine(module_a, module_b)
    results = {item["component_id"]: item for item in report["component_results"]}
    assert results["C001"]["final_decision"] == "RETEST_REQUIRED"
    assert "MISSING_MODULE_B_RESULT" in results["C001"]["reason_codes"]
    assert results["C002"]["final_decision"] == "RETEST_REQUIRED"
    assert "MISSING_MODULE_A_RESULT" in results["C002"]["reason_codes"]


def test_unknown_module_outcome_requires_retest_instead_of_silent_clearance():
    report = combine_reports(
        *_reports(
            _module_a_component(status="UNKNOWN"),
            _module_b_component(decision="CONTINUE_SCREENING"),
        )
    )
    result = report["component_results"][0]
    assert result["final_decision"] == "RETEST_REQUIRED"
    assert "INVALID_MODULE_A_STATUS" in result["reason_codes"]


def test_mismatched_lots_and_checkpoints_are_rejected():
    module_a, module_b = _reports(_module_a_component(), _module_b_component())
    module_b["lot_id"] = "OTHER_LOT"
    with pytest.raises(ValueError, match="lot mismatch"):
        combine_reports(module_a, module_b)

    module_b["lot_id"] = "LOT_TEST"
    module_b["input_as_of_h"] = 96.0
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        combine_reports(module_a, module_b)


def test_real_module_reports_combine_at_24h(
    trained_engine,
    trained_module_b_engine,
    test_lot,
):
    module_a = trained_engine.score_lot(test_lot, as_of_h=24.0)
    module_b = trained_module_b_engine.forecast_lot(test_lot)
    report = combine_reports(module_a, module_b)

    assert report["lot_id"] == "TEST_LOT_001"
    assert report["as_of_h"] == 24.0
    assert report["forecast_target_h"] == 168.0
    assert len(report["component_results"]) == test_lot["component_id"].nunique()
    assert sum(report["summary"]["decision_counts"].values()) == len(
        report["component_results"]
    )


def test_ensemble_partition_evaluation_reports_final_detection_metrics(
    trained_engine,
    trained_module_b_engine,
    test_lot,
):
    labelled_lot = test_lot.copy()
    labelled_component = str(labelled_lot["component_id"].iloc[0])
    labelled_lot.loc[
        labelled_lot["component_id"] == labelled_component,
        "is_anomaly",
    ] = True
    labelled_lot.loc[
        labelled_lot["component_id"] == labelled_component,
        "defect_type",
    ] = "labelled_test_defect"
    report = evaluate_partition(
        trained_engine,
        trained_module_b_engine,
        labelled_lot,
        as_of_h=24.0,
    )

    assert report["component_count"] == labelled_lot["component_id"].nunique()
    assert report["true_positives"] + report["false_negatives"] > 0
    assert report["false_positives"] + report["true_negatives"] > 0
    assert 0.0 <= report["defect_recall"] <= 1.0
    assert 0.0 <= report["hard_reject_recall"] <= report["defect_recall"]
    assert sum(report["decision_counts"].values()) == report["component_count"]
