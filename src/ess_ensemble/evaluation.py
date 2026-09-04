"""Lot-safe evaluation of the combined Module A and Module B disposition."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from ess_module_a.config import ModuleAConfig, default_config as module_a_defaults
from ess_module_a.engine import ModuleAEngine
from ess_module_a.evaluation import split_lots
from ess_module_b.config import ModuleBConfig, default_config as module_b_defaults
from ess_module_b.engine import ModuleBEngine

from .engine import RULE_VERSION, EnsembleEngine


ACTION_DECISIONS = {"MONITOR", "RETEST_REQUIRED", "REJECT_EARLY"}
MODULE_A_ACTION_STATUSES = {"MONITOR", "QUARANTINE", "STATIC_FAIL", "RETEST_REQUIRED"}
MODULE_B_ACTION_DECISIONS = {"EARLY_REJECT", "STATIC_FAIL", "RETEST_REQUIRED"}


def _component_labels(measurements: pd.DataFrame) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for component_id, group in measurements.groupby("component_id", sort=False):
        anomaly_values = group.get("is_anomaly", pd.Series(False, index=group.index))
        is_anomaly = bool(anomaly_values.fillna(False).astype(bool).any())
        defect_values = group.get("defect_type", pd.Series("normal", index=group.index))
        defect_types = sorted(
            {
                str(value)
                for value in defect_values.dropna().unique()
                if str(value).lower() not in {"normal", "nan"}
            }
        )
        labels[str(component_id)] = {
            "is_anomaly": is_anomaly,
            "defect_type": "+".join(defect_types) if defect_types else "normal",
        }
    return labels


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_partition(
    module_a_engine: ModuleAEngine,
    module_b_engine: ModuleBEngine,
    measurements: pd.DataFrame,
    *,
    as_of_h: float = 24.0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    lot_recalls: list[float] = []
    decision_counts: Counter[str] = Counter()
    evidence_patterns: Counter[str] = Counter()

    for lot_id, lot in measurements.groupby("lot_id", sort=False):
        module_a_report = module_a_engine.score_lot(lot, as_of_h=as_of_h)
        module_b_report = module_b_engine.forecast_lot(lot)
        ensemble_report = EnsembleEngine().combine(module_a_report, module_b_report)
        labels = _component_labels(lot)
        lot_records: list[dict[str, Any]] = []

        for result in ensemble_report["component_results"]:
            component_id = result["component_id"]
            label = labels[component_id]
            action = result["final_decision"] in ACTION_DECISIONS
            hard_reject = result["final_decision"] == "REJECT_EARLY"
            record = {
                "lot_id": str(lot_id),
                "component_id": component_id,
                "actual": bool(label["is_anomaly"]),
                "defect_type": label["defect_type"],
                "action": action,
                "hard_reject": hard_reject,
                "module_a_action": result["module_a_status"]
                in MODULE_A_ACTION_STATUSES,
                "module_b_action": result["module_b_decision"]
                in MODULE_B_ACTION_DECISIONS,
                "final_decision": result["final_decision"],
                "evidence_pattern": result["evidence_pattern"],
                "risk_score": float(result["ensemble_risk_score"]),
            }
            records.append(record)
            lot_records.append(record)
            decision_counts[result["final_decision"]] += 1
            evidence_patterns[result["evidence_pattern"]] += 1

        positives = [record for record in lot_records if record["actual"]]
        if positives:
            lot_recalls.append(
                sum(record["action"] for record in positives) / len(positives)
            )

    tp = sum(record["actual"] and record["action"] for record in records)
    fn = sum(record["actual"] and not record["action"] for record in records)
    fp = sum(not record["actual"] and record["action"] for record in records)
    tn = sum(not record["actual"] and not record["action"] for record in records)
    hard_reject_tp = sum(
        record["actual"] and record["hard_reject"] for record in records
    )
    actual_positive_count = tp + fn

    by_defect: dict[str, dict[str, int | float | None]] = {}
    defect_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["actual"]:
            defect_groups[record["defect_type"]].append(record)
    for defect_type, items in sorted(defect_groups.items()):
        detected = sum(item["action"] for item in items)
        rejected = sum(item["hard_reject"] for item in items)
        by_defect[defect_type] = {
            "count": len(items),
            "detected": detected,
            "action_recall": _safe_ratio(detected, len(items)),
            "hard_rejected": rejected,
            "hard_reject_recall": _safe_ratio(rejected, len(items)),
        }

    return {
        "as_of_h": as_of_h,
        "forecast_target_h": module_b_engine.config.target_h,
        "component_count": len(records),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "defect_recall": _safe_ratio(tp, actual_positive_count),
        "hard_reject_true_positives": hard_reject_tp,
        "hard_reject_recall": _safe_ratio(hard_reject_tp, actual_positive_count),
        "precision": _safe_ratio(tp, tp + fp),
        "false_flag_rate": _safe_ratio(fp, fp + tn),
        "weighted_error_cost": module_b_engine.config.false_negative_cost * fn + fp,
        "worst_lot_recall": min(lot_recalls) if lot_recalls else None,
        "false_negative_components": [
            record["component_id"]
            for record in records
            if record["actual"] and not record["action"]
        ],
        "module_a_detected_defects": sum(
            record["actual"] and record["module_a_action"] for record in records
        ),
        "module_b_detected_defects": sum(
            record["actual"] and record["module_b_action"] for record in records
        ),
        "recall_by_defect_type": by_defect,
        "decision_counts": dict(decision_counts),
        "evidence_pattern_counts": dict(evidence_patterns),
    }


def evaluate_lot_safe(
    measurements: pd.DataFrame,
    module_a_config: ModuleAConfig | None = None,
    module_b_config: ModuleBConfig | None = None,
    *,
    seed: int = 170,
    as_of_h: float = 24.0,
) -> dict[str, Any]:
    partitions = split_lots(measurements, seed=seed)
    module_a_engine = ModuleAEngine(module_a_config or module_a_defaults())
    module_b_engine = ModuleBEngine(module_b_config or module_b_defaults())
    module_a_engine.fit(partitions["train"])
    module_b_engine.fit(partitions["train"])

    return {
        "ensemble_version": "0.1.0",
        "rule_version": RULE_VERSION,
        "split_lots": {
            name: sorted(str(value) for value in frame["lot_id"].unique())
            for name, frame in partitions.items()
        },
        "module_a_model_info": module_a_engine.model_info(),
        "module_b_model_info": module_b_engine.model_info(),
        "validation": evaluate_partition(
            module_a_engine,
            module_b_engine,
            partitions["validation"],
            as_of_h=as_of_h,
        ),
        "test": evaluate_partition(
            module_a_engine,
            module_b_engine,
            partitions["test"],
            as_of_h=as_of_h,
        ),
    }
