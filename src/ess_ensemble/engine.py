"""Explainable, safety-first decision fusion for ESS Modules A and B."""

from __future__ import annotations

from collections import Counter
from enum import Enum
import math
from typing import Any


class EnsembleDecision(str, Enum):
    """Final early-screening outcomes produced by the ensemble."""

    CONTINUE_SCREENING = "CONTINUE_SCREENING"
    MONITOR = "MONITOR"
    RETEST_REQUIRED = "RETEST_REQUIRED"
    REJECT_EARLY = "REJECT_EARLY"


DECISION_PRIORITY = {
    EnsembleDecision.CONTINUE_SCREENING: 0,
    EnsembleDecision.MONITOR: 1,
    EnsembleDecision.RETEST_REQUIRED: 2,
    EnsembleDecision.REJECT_EARLY: 3,
}

MODULE_A_REJECT_STATUSES = {"QUARANTINE", "STATIC_FAIL"}
MODULE_B_REJECT_DECISIONS = {"EARLY_REJECT", "STATIC_FAIL"}
MODULE_A_VALID_STATUSES = {
    "NORMAL",
    "MONITOR",
    "QUARANTINE",
    "STATIC_FAIL",
    "RETEST_REQUIRED",
}
MODULE_B_VALID_DECISIONS = {
    "CONTINUE_SCREENING",
    "EARLY_REJECT",
    "STATIC_FAIL",
    "RETEST_REQUIRED",
}
RULE_VERSION = "1.0.0"


def _risk(value: Any, *, missing: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return min(1.0, max(0.0, number))


def _index_components(
    report: dict[str, Any],
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in report.get("component_results", []):
        component_id = str(item.get("component_id", ""))
        if not component_id:
            raise ValueError(f"{source} contains a component result without component_id")
        if component_id in indexed:
            raise ValueError(f"{source} contains duplicate component_id {component_id}")
        indexed[component_id] = item
    return indexed


def _validate_reports(module_a_report: dict[str, Any], module_b_report: dict[str, Any]) -> str | None:
    lot_a = module_a_report.get("lot_id")
    lot_b = module_b_report.get("lot_id")
    if lot_a is not None and lot_b is not None and str(lot_a) != str(lot_b):
        raise ValueError(f"Module report lot mismatch: {lot_a} != {lot_b}")

    as_of_a = module_a_report.get("as_of_h")
    as_of_b = module_b_report.get("input_as_of_h")
    if as_of_a is not None and as_of_b is not None:
        if not math.isclose(float(as_of_a), float(as_of_b), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Module report checkpoint mismatch: {as_of_a} h != {as_of_b} h"
            )
    return str(lot_a if lot_a is not None else lot_b) if lot_a is not None or lot_b is not None else None


def _module_pattern(a_rejects: bool, b_rejects: bool) -> str:
    if a_rejects and b_rejects:
        return "BOTH_MODULES_REJECT"
    if a_rejects:
        return "MODULE_A_ONLY_REJECT"
    if b_rejects:
        return "MODULE_B_ONLY_REJECT"
    return "NO_HARD_REJECT"


def _explanation(
    decision: EnsembleDecision,
    pattern: str,
    module_a_status: str | None,
    module_b_decision: str | None,
) -> str:
    if pattern == "BOTH_MODULES_REJECT":
        return (
            "Both modules independently require rejection: Module A found an observed "
            "dynamic/static anomaly and Module B forecasts unsafe 168-hour behaviour."
        )
    if pattern == "MODULE_A_ONLY_REJECT":
        return (
            "Reject early because Module A found an observed dynamic/static anomaly, "
            "even though Module B did not issue a hard forecast rejection."
        )
    if pattern == "MODULE_B_ONLY_REJECT":
        return (
            "Reject early because Module B forecasts unsafe 168-hour drift, even though "
            "Module A has not observed a hard anomaly by 24 hours."
        )
    if decision is EnsembleDecision.RETEST_REQUIRED:
        return (
            "Do not clear the component: at least one module could not produce reliable "
            "evidence and requires a retest."
        )
    if decision is EnsembleDecision.MONITOR:
        return (
            "Continue under enhanced monitoring: Module A found warning-level observed "
            "evidence while Module B did not issue a hard rejection."
        )
    if module_a_status is None or module_b_decision is None:
        return "A module result is missing; the component cannot be cleared."
    return (
        "Both modules are clear at the 24-hour decision point: no observed hard anomaly "
        "and no unsafe 168-hour forecast trigger were found. Continue screening; this is "
        "not final flight acceptance."
    )


class EnsembleEngine:
    """Fuse precomputed Module A and Module B reports without hiding provenance."""

    def combine(
        self,
        module_a_report: dict[str, Any],
        module_b_report: dict[str, Any],
    ) -> dict[str, Any]:
        lot_id = _validate_reports(module_a_report, module_b_report)
        module_a = _index_components(module_a_report, source="Module A")
        module_b = _index_components(module_b_report, source="Module B")
        component_ids = sorted(set(module_a) | set(module_b))

        component_results = [
            self._combine_component(component_id, module_a.get(component_id), module_b.get(component_id))
            for component_id in component_ids
        ]
        counts = Counter(item["final_decision"] for item in component_results)
        patterns = Counter(item["evidence_pattern"] for item in component_results)
        return {
            "lot_id": lot_id,
            "as_of_h": module_a_report.get("as_of_h", module_b_report.get("input_as_of_h")),
            "forecast_target_h": module_b_report.get("target_h"),
            "ensemble_version": "0.1.0",
            "rule_version": RULE_VERSION,
            "component_results": component_results,
            "summary": {
                "component_count": len(component_results),
                "decision_counts": dict(counts),
                "evidence_pattern_counts": dict(patterns),
                "qa_action_count": sum(
                    item["final_decision"]
                    in {
                        EnsembleDecision.REJECT_EARLY.value,
                        EnsembleDecision.RETEST_REQUIRED.value,
                    }
                    for item in component_results
                ),
                "hard_reject_count": counts[EnsembleDecision.REJECT_EARLY.value],
                "module_a_hard_reject_count": sum(
                    item["module_a_status"] in MODULE_A_REJECT_STATUSES
                    for item in component_results
                ),
                "module_b_hard_reject_count": sum(
                    item["module_b_decision"] in MODULE_B_REJECT_DECISIONS
                    for item in component_results
                ),
                "consensus_reject_count": patterns["BOTH_MODULES_REJECT"],
            },
            "module_a_validation_issues": module_a_report.get("validation_issues", []),
            "module_b_validation_issues": module_b_report.get("validation_issues", []),
        }

    @staticmethod
    def _combine_component(
        component_id: str,
        module_a: dict[str, Any] | None,
        module_b: dict[str, Any] | None,
    ) -> dict[str, Any]:
        module_a_status = str(module_a.get("status")) if module_a is not None else None
        module_b_decision = str(module_b.get("decision")) if module_b is not None else None
        a_rejects = module_a_status in MODULE_A_REJECT_STATUSES
        b_rejects = module_b_decision in MODULE_B_REJECT_DECISIONS
        a_is_valid = module_a_status in MODULE_A_VALID_STATUSES
        b_is_valid = module_b_decision in MODULE_B_VALID_DECISIONS
        pattern = _module_pattern(a_rejects, b_rejects)

        if a_rejects or b_rejects:
            decision = EnsembleDecision.REJECT_EARLY
        elif module_a is None or module_b is None or not a_is_valid or not b_is_valid:
            decision = EnsembleDecision.RETEST_REQUIRED
        elif module_a_status == "RETEST_REQUIRED" or module_b_decision == "RETEST_REQUIRED":
            decision = EnsembleDecision.RETEST_REQUIRED
        elif module_a_status == "MONITOR":
            decision = EnsembleDecision.MONITOR
        else:
            decision = EnsembleDecision.CONTINUE_SCREENING

        risk_a = _risk(module_a.get("risk_score"), missing=1.0) if module_a else 1.0
        risk_b = _risk(module_b.get("risk_score"), missing=1.0) if module_b else 1.0
        ensemble_risk = min(1.0, 1.0 - (1.0 - risk_a) * (1.0 - risk_b))

        reasons: list[str] = []
        if module_a is None:
            reasons.append("MISSING_MODULE_A_RESULT")
        elif not a_is_valid:
            reasons.append("INVALID_MODULE_A_STATUS")
        if module_b is None:
            reasons.append("MISSING_MODULE_B_RESULT")
        elif not b_is_valid:
            reasons.append("INVALID_MODULE_B_DECISION")
        if a_rejects:
            reasons.append(f"MODULE_A_{module_a_status}")
        elif module_a_status == "RETEST_REQUIRED":
            reasons.append("MODULE_A_RETEST_REQUIRED")
        elif module_a_status == "MONITOR":
            reasons.append("MODULE_A_MONITOR")
        if b_rejects:
            reasons.append(f"MODULE_B_{module_b_decision}")
        elif module_b_decision == "RETEST_REQUIRED":
            reasons.append("MODULE_B_RETEST_REQUIRED")
        if pattern == "BOTH_MODULES_REJECT":
            reasons.append("CROSS_MODULE_REJECT_CONSENSUS")
        if not reasons:
            reasons.append("NO_ENSEMBLE_TRIGGER")

        return {
            "component_id": component_id,
            "final_decision": decision.value,
            "qa_action_required": decision
            in {EnsembleDecision.REJECT_EARLY, EnsembleDecision.RETEST_REQUIRED},
            "flagged_for_early_rejection": decision is EnsembleDecision.REJECT_EARLY,
            "ensemble_risk_score": ensemble_risk,
            "evidence_pattern": pattern,
            "module_a_status": module_a_status,
            "module_a_risk_score": risk_a,
            "module_a_highest_risk_parameter": module_a.get("highest_risk_parameter")
            if module_a
            else None,
            "module_a_reason_codes": list(module_a.get("reason_codes") or [])
            if module_a
            else [],
            "module_b_decision": module_b_decision,
            "module_b_risk_score": risk_b,
            "module_b_highest_risk_parameter": module_b.get("highest_risk_parameter")
            if module_b
            else None,
            "module_b_predicted_value_168h": module_b.get("predicted_value_168h")
            if module_b
            else None,
            "module_b_reason_codes": list(module_b.get("reason_codes") or [])
            if module_b
            else [],
            "reason_codes": reasons,
            "explanation": _explanation(
                decision,
                pattern,
                module_a_status,
                module_b_decision,
            ),
            "rule_version": RULE_VERSION,
        }


def combine_reports(
    module_a_report: dict[str, Any],
    module_b_report: dict[str, Any],
) -> dict[str, Any]:
    """Convenience function for one-off report fusion."""

    return EnsembleEngine().combine(module_a_report, module_b_report)
