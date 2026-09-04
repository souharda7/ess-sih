"""Public reference fitting and lot-scoring engine."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ModuleAConfig, default_config
from .detectors import evaluate_robust_rules, static_check
from .features import build_feature_frame
from .models import QAStatus, ReferenceProfile, STATUS_PRIORITY, worst_status
from .reference import fit_reference_profile, load_reference, save_reference
from .statistics import direction_risk, percentile_of_scores, robust_z
from .validation import DataValidationError, validate_measurements


def _number(value: Any) -> float | None:
    if value is None or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _stats_summary(value: Any) -> dict[str, Any] | None:
    return value.to_summary() if value is not None else None


def fit_reference(
    training_measurements: pd.DataFrame,
    parameter_config: ModuleAConfig | None = None,
) -> ReferenceProfile:
    return fit_reference_profile(training_measurements, parameter_config or default_config())


def score_lot(
    lot_measurements: pd.DataFrame,
    reference_profile: ReferenceProfile,
    parameter_config: ModuleAConfig | None = None,
    *,
    as_of_h: float | None = None,
) -> dict[str, Any]:
    engine = ModuleAEngine(parameter_config or default_config(), reference_profile)
    return engine.score_lot(lot_measurements, as_of_h=as_of_h)


class ModuleAEngine:
    def __init__(
        self,
        config: ModuleAConfig | None = None,
        reference_profile: ReferenceProfile | None = None,
    ) -> None:
        self.config = config or default_config()
        self.reference = reference_profile

    def fit(self, measurements: pd.DataFrame) -> ReferenceProfile:
        self.reference = fit_reference_profile(measurements, self.config)
        return self.reference

    def save(self, path: str | Path) -> None:
        if self.reference is None:
            raise RuntimeError("No reference profile has been fitted")
        save_reference(self.reference, path)

    @classmethod
    def load(
        cls, path: str | Path, config: ModuleAConfig | None = None
    ) -> "ModuleAEngine":
        return cls(config=config or default_config(), reference_profile=load_reference(path))

    def model_info(self) -> dict[str, Any]:
        reference = self.reference
        return {
            "module_version": "0.1.0",
            "config_version": self.config.version,
            "reference_loaded": reference is not None,
            "reference_version": reference.version if reference else None,
            "reference_created_at_utc": reference.created_at_utc if reference else None,
            "training_lot_count": len(reference.training_lot_ids) if reference else 0,
            "mahalanobis_context_count": len(reference.mahalanobis_models) if reference else 0,
        }

    def score_lot(
        self, measurements: pd.DataFrame, *, as_of_h: float | None = None
    ) -> dict[str, Any]:
        if self.reference is None:
            raise RuntimeError("A fitted historical reference is required before scoring")
        if self.reference.config_version != self.config.version:
            raise RuntimeError(
                "Reference/config version mismatch: "
                f"{self.reference.config_version} != {self.config.version}"
            )

        validation = validate_measurements(
            measurements, self.config, as_of_h=as_of_h, require_single_lot=True
        )
        frame = validation.measurements
        feature_frame = build_feature_frame(frame, self.config, self.reference)
        feature_frame = self._attach_model_scores(feature_frame)
        feature_lookup = {
            int(row["row_index"]): row for row in feature_frame.to_dict("records")
        }

        parameter_results: list[dict[str, Any]] = []
        scored_frame = frame.loc[frame["time_h"].notna() & (frame["time_h"] <= validation.as_of_h)]
        for index, measurement in scored_frame.iterrows():
            key = (
                str(measurement["component_id"]),
                str(measurement["parameter"]),
                str(measurement["test_condition_id"]),
            )
            feature = feature_lookup.get(int(index))
            result = self._score_parameter_row(
                measurement.to_dict(), feature, validation.series_issues.get(key, [])
            )
            parameter_results.append(result)

        component_results = self._aggregate_components(parameter_results)
        lot_alerts = self._lot_alerts(feature_frame)
        return {
            "lot_id": validation.lot_id,
            "as_of_h": validation.as_of_h,
            "module_version": "0.1.0",
            "config_version": self.config.version,
            "reference_version": self.reference.version,
            "parameter_results": parameter_results,
            "component_results": component_results,
            "lot_alerts": lot_alerts,
            "validation_issues": [issue.to_dict() for issue in validation.issues],
        }

    def _attach_model_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        result = features.copy()
        result["mahalanobis_score"] = np.nan
        result["mahalanobis_percentile"] = np.nan
        if result.empty:
            return result

        context_columns = ["part_number", "time_h", "test_condition_id"]
        for context, group in result.groupby(context_columns, sort=False):
            artifact = self.reference.mahalanobis_models.get(tuple(context)) if self.reference else None
            if artifact is None:
                continue
            for component_id, component_group in group.groupby("component_id", sort=False):
                values = {
                    str(row["parameter"]): row["historical_z_signed"]
                    for row in component_group.to_dict("records")
                }
                if any(parameter not in values for parameter in artifact.parameters):
                    continue
                vector = np.asarray([[values[parameter] for parameter in artifact.parameters]], dtype=float)
                if not np.isfinite(vector).all():
                    continue
                score = float(artifact.estimator.mahalanobis(vector)[0])
                percentile = percentile_of_scores(score, artifact.training_scores)
                indexes = component_group.index
                result.loc[indexes, "mahalanobis_score"] = score
                result.loc[indexes, "mahalanobis_percentile"] = percentile
        return result

    def _score_parameter_row(
        self,
        measurement: dict[str, Any],
        feature: dict[str, Any] | None,
        series_issues: list[str],
    ) -> dict[str, Any]:
        parameter_name = str(measurement.get("parameter"))
        parameter = self.config.parameters.get(parameter_name)
        base = {
            "component_id": str(measurement.get("component_id")),
            "lot_id": str(measurement.get("lot_id")),
            "part_number": str(measurement.get("part_number")),
            "parameter": parameter_name,
            "time_h": _number(measurement.get("time_h")),
            "test_condition_id": str(measurement.get("test_condition_id")),
            "normalized_value": _number(measurement.get("normalized_value")),
            "unit": parameter.canonical_unit if parameter else str(measurement.get("unit")),
            "data_quality_status": "VALID" if bool(measurement.get("_valid")) else "INVALID",
        }
        row_issues = list(measurement.get("_issues") or [])
        quality_reasons = list(dict.fromkeys([*row_issues, *series_issues]))

        if parameter is None or feature is None or not bool(measurement.get("_valid")):
            return {
                **base,
                "static_status": "NOT_EVALUATED",
                "static_margin": None,
                "lot_statistics": None,
                "historical_statistics": None,
                "robust_z_lot": None,
                "robust_z_historical": None,
                "lot_percentile": None,
                "historical_percentile": None,
                "percentage_from_lot_median": None,
                "slope": None,
                "slope_from_zero": None,
                "slope_robust_z_lot": None,
                "slope_robust_z_historical": None,
                "mahalanobis_score": None,
                "mahalanobis_percentile": None,
                "status": QAStatus.RETEST_REQUIRED.value,
                "risk_score": 1.0,
                "reason_codes": quality_reasons or ["MEASUREMENT_INVALID"],
            }

        value = float(measurement["normalized_value"])
        static_failed, static_margin, static_reasons = static_check(value, parameter)
        rules = evaluate_robust_rules(feature, parameter, self.config)
        reasons = [*static_reasons, *rules["reason_codes"]]
        warning_categories = set(rules["warning_categories"])
        severe_categories = set(rules["severe_categories"])

        mahalanobis_percentile = _number(feature.get("mahalanobis_percentile"))
        if mahalanobis_percentile is not None:
            if mahalanobis_percentile >= self.config.tail_severe_percentile:
                severe_categories.add("mahalanobis")
                reasons.append("MAHALANOBIS_SEVERE")
            elif mahalanobis_percentile >= self.config.tail_warning_percentile:
                warning_categories.add("mahalanobis")
                reasons.append("MAHALANOBIS_WARNING")

        references_missing = feature.get("lot_stats") is None and feature.get("historical_stats") is None
        if references_missing:
            quality_reasons.append("REFERENCE_INSUFFICIENT")

        if static_failed:
            status = QAStatus.STATIC_FAIL
        elif quality_reasons or references_missing:
            status = QAStatus.RETEST_REQUIRED
        else:
            slope_sources = {"slope_lot", "slope_historical"}
            evidence_weight = 2 * len(severe_categories) + len(warning_categories)
            corroborated_slope = bool(severe_categories & slope_sources) and len(
                (severe_categories | warning_categories) - slope_sources
            ) >= 3
            if rules["extreme"] or evidence_weight >= 8 or corroborated_slope:
                status = QAStatus.QUARANTINE
            elif evidence_weight >= 6:
                status = QAStatus.MONITOR
            else:
                status = QAStatus.NORMAL

        reasons.extend(quality_reasons)
        risk_values = [
            max(0.0, _number(feature.get("risk_lot_z")) or 0.0) / self.config.robust_z_extreme,
            max(0.0, _number(feature.get("risk_historical_z")) or 0.0)
            / self.config.robust_z_extreme,
            (mahalanobis_percentile or 0.0) / 100.0,
        ]
        risk_score = min(1.0, max(risk_values))
        if static_failed or quality_reasons:
            risk_score = 1.0

        return {
            **base,
            "static_status": "FAIL" if static_failed else "PASS",
            "static_margin": _number(static_margin),
            "lot_statistics": _stats_summary(feature.get("lot_stats")),
            "historical_statistics": _stats_summary(feature.get("historical_stats")),
            "robust_z_lot": _number(feature.get("lot_z")),
            "robust_z_historical": _number(feature.get("historical_z")),
            "lot_percentile": _number(feature.get("lot_percentile")),
            "historical_percentile": _number(feature.get("historical_percentile")),
            "percentage_from_lot_median": _number(feature.get("percentage_from_lot_median")),
            "lot_shift_robust_z": _number(feature.get("lot_shift_z")),
            "slope": _number(feature.get("slope")),
            "slope_from_zero": _number(feature.get("slope_from_zero")),
            "slope_start_h": _number(feature.get("slope_start_h")),
            "slope_robust_z_lot": _number(feature.get("slope_lot_z")),
            "slope_robust_z_historical": _number(feature.get("slope_historical_z")),
            "mahalanobis_score": _number(feature.get("mahalanobis_score")),
            "mahalanobis_percentile": mahalanobis_percentile,
            "status": status.value,
            "risk_score": risk_score,
            "warning_categories": sorted(warning_categories),
            "severe_categories": sorted(severe_categories),
            "reason_codes": list(dict.fromkeys(reasons)),
        }

    def _aggregate_components(
        self, parameter_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in parameter_results:
            grouped[result["component_id"]].append(result)
        output = []
        for component_id, results in sorted(grouped.items()):
            status = worst_status([QAStatus(item["status"]) for item in results])
            highest = max(
                results,
                key=lambda item: (
                    STATUS_PRIORITY[QAStatus(item["status"])],
                    item.get("risk_score", 0.0),
                ),
            )
            output.append(
                {
                    "component_id": component_id,
                    "status": status.value,
                    "risk_score": max(float(item.get("risk_score", 0.0)) for item in results),
                    "highest_risk_parameter": highest["parameter"],
                    "highest_risk_checkpoint_h": highest["time_h"],
                    "reason_codes": sorted(
                        {code for item in results for code in item.get("reason_codes", [])}
                    ),
                    "config_version": self.config.version,
                    "reference_version": self.reference.version if self.reference else None,
                }
            )
        return output

    def _lot_alerts(self, features: pd.DataFrame) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        if features.empty:
            return alerts
        group_columns = ["part_number", "parameter", "time_h", "test_condition_id"]
        for context, group in features.groupby(group_columns, sort=False):
            risk_values = group["risk_lot_shift_z"].dropna()
            if risk_values.empty:
                continue
            risk = float(risk_values.iloc[0])
            if risk < self.config.robust_z_warning:
                continue
            alerts.append(
                {
                    "type": "WHOLE_LOT_SHIFT",
                    "severity": "SEVERE" if risk >= self.config.robust_z_severe else "WARNING",
                    "part_number": str(context[0]),
                    "parameter": str(context[1]),
                    "time_h": float(context[2]),
                    "test_condition_id": str(context[3]),
                    "direction_aware_robust_z": risk,
                    "reason_code": "WHOLE_LOT_SHIFT_Z",
                }
            )

        alerts.extend(self._equipment_shift_alerts(features, "tester_id", "TESTER_SHIFT"))
        alerts.extend(self._equipment_shift_alerts(features, "chamber_id", "CHAMBER_SHIFT"))
        return alerts

    def _equipment_shift_alerts(
        self, features: pd.DataFrame, field: str, alert_type: str
    ) -> list[dict[str, Any]]:
        if field not in features.columns:
            return []
        alerts: list[dict[str, Any]] = []
        columns = ["lot_id", "part_number", "parameter", "time_h", "test_condition_id", field]
        for context, group in features.dropna(subset=[field]).groupby(columns, sort=False):
            if len(group) < 5:
                continue
            lot_stats = group.iloc[0].get("lot_stats")
            parameter = self.config.parameters[str(context[2])]
            if lot_stats is None:
                continue
            subgroup_median = float(group["normalized_value"].median())
            z_value = robust_z(subgroup_median, lot_stats, self.config.epsilon)
            risk = direction_risk(z_value, parameter.danger_direction)
            if risk >= self.config.robust_z_warning:
                alerts.append(
                    {
                        "type": alert_type,
                        "severity": "SEVERE" if risk >= self.config.robust_z_severe else "WARNING",
                        "part_number": str(context[1]),
                        "parameter": str(context[2]),
                        "time_h": float(context[3]),
                        "test_condition_id": str(context[4]),
                        field: str(context[5]),
                        "direction_aware_robust_z": risk,
                        "reason_code": alert_type,
                    }
                )
        return alerts
