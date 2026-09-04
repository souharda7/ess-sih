"""Public fitting and explainable 168-hour forecasting engine."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ess_module_a.models import DangerDirection
from ess_module_a.units import convert_value

from .config import ModuleBConfig, default_config
from .data import prepare_series
from .features import (
    FEATURE_NAMES,
    direction_aware_slope,
    engineer_features,
    linear_extrapolation_prediction,
    persistence_prediction,
)
from .models import (
    DECISION_PRIORITY,
    DriftDecision,
    DriftPredictorArtifact,
    FittedDriftModel,
    worst_decision,
)
from .training import fit_predictor_artifact, load_artifact, save_artifact


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _relative_improvement(model_mae: float, baseline_mae: float) -> float | None:
    if baseline_mae <= 0:
        return None
    return float(100.0 * (baseline_mae - model_mae) / baseline_mae)


def fit_predictor(
    training_measurements: pd.DataFrame,
    config: ModuleBConfig | None = None,
) -> DriftPredictorArtifact:
    return fit_predictor_artifact(training_measurements, config or default_config())


def forecast_lot(
    early_measurements: pd.DataFrame,
    predictor: DriftPredictorArtifact,
    config: ModuleBConfig | None = None,
) -> dict[str, Any]:
    return ModuleBEngine(config or default_config(), predictor).forecast_lot(early_measurements)


class ModuleBEngine:
    def __init__(
        self,
        config: ModuleBConfig | None = None,
        artifact: DriftPredictorArtifact | None = None,
    ) -> None:
        self.config = config or default_config()
        self.artifact = artifact

    def fit(self, measurements: pd.DataFrame) -> DriftPredictorArtifact:
        self.artifact = fit_predictor_artifact(measurements, self.config)
        return self.artifact

    def save(self, path: str | Path) -> None:
        if self.artifact is None:
            raise RuntimeError("No Module B predictor has been fitted")
        save_artifact(self.artifact, path)

    @classmethod
    def load(
        cls, path: str | Path, config: ModuleBConfig | None = None
    ) -> "ModuleBEngine":
        return cls(config=config or default_config(), artifact=load_artifact(path))

    def model_info(self) -> dict[str, Any]:
        artifact = self.artifact
        selected = []
        contexts = []
        if artifact is not None:
            for key, model in sorted(artifact.context_models.items()):
                selected.append(model.name)
                contexts.append(
                    {
                        "part_number": key[0],
                        "parameter": key[1],
                        "test_condition_id": key[2],
                        "model": model.name,
                        "training_samples": model.training_samples,
                        "training_lots": model.training_lots,
                        "cv_mae": model.cv_mae,
                        "persistence_mae": model.persistence_mae,
                        "linear_extrapolation_mae": model.linear_extrapolation_mae,
                    }
                )
        return {
            "module_version": "0.1.0",
            "config_version": self.config.version,
            "artifact_loaded": artifact is not None,
            "artifact_version": artifact.version if artifact else None,
            "artifact_created_at_utc": artifact.created_at_utc if artifact else None,
            "training_lot_count": len(artifact.training_lot_ids) if artifact else 0,
            "context_model_count": len(artifact.context_models) if artifact else 0,
            "parameter_fallback_model_count": len(artifact.parameter_models) if artifact else 0,
            "selected_model_counts": dict(Counter(selected)),
            "input_checkpoints_h": [self.config.baseline_h, self.config.early_h],
            "target_checkpoint_h": self.config.target_h,
            "contexts": contexts,
        }

    def forecast_lot(self, measurements: pd.DataFrame) -> dict[str, Any]:
        if self.artifact is None:
            raise RuntimeError("A fitted Module B predictor is required before forecasting")
        if self.artifact.config_version != self.config.version:
            raise RuntimeError(
                "Predictor/config version mismatch: "
                f"{self.artifact.config_version} != {self.config.version}"
            )
        prepared = prepare_series(
            measurements,
            self.config,
            include_target=False,
            require_single_lot=True,
        )
        predictions = [self._forecast_series(row) for _, row in prepared.series.iterrows()]
        component_results = self._aggregate_components(predictions)
        lot_ids = sorted(str(value) for value in prepared.series.get("lot_id", pd.Series()).unique())
        return {
            "lot_id": lot_ids[0] if len(lot_ids) == 1 else None,
            "input_as_of_h": self.config.early_h,
            "target_h": self.config.target_h,
            "module_version": "0.1.0",
            "config_version": self.config.version,
            "artifact_version": self.artifact.version,
            "prediction_results": predictions,
            "component_results": component_results,
            "validation_issues": prepared.validation_issues,
            "ignored_measurement_count": prepared.ignored_measurement_count,
        }

    def _forecast_series(self, series: pd.Series) -> dict[str, Any]:
        parameter_name = str(series.get("parameter"))
        parameter = self.config.parameters.get(parameter_name)
        base = {
            "component_id": str(series.get("component_id")),
            "lot_id": str(series.get("lot_id")),
            "part_number": str(series.get("part_number")),
            "parameter": parameter_name,
            "test_condition_id": str(series.get("test_condition_id")),
            "unit": series.get("unit"),
            "value_0h": _number(series.get("value_0h")),
            "value_24h": _number(series.get("value_24h")),
        }
        quality_codes = list(series.get("reason_codes") or [])
        if parameter is None:
            quality_codes.append("UNKNOWN_PARAMETER")
        if quality_codes or parameter is None or base["value_0h"] is None or base["value_24h"] is None:
            return self._unscored_result(base, quality_codes or ["INCOMPLETE_EARLY_SERIES"])

        context = (
            str(series["part_number"]),
            parameter_name,
            str(series["test_condition_id"]),
        )
        model = self.artifact.context_models.get(context) if self.artifact else None
        model_scope = "exact_context"
        reasons: list[str] = []
        if model is None and self.artifact is not None:
            model = self.artifact.parameter_models.get(parameter_name)
            model_scope = "parameter_fallback"
            if model is not None:
                reasons.append("PARAMETER_FALLBACK_MODEL")
        if model is None:
            return self._unscored_result(base, ["NO_COMPATIBLE_MODEL"])

        row_frame = pd.DataFrame([series.to_dict()])
        feature_frame = engineer_features(
            row_frame,
            baseline_h=self.config.baseline_h,
            early_h=self.config.early_h,
            epsilon=self.config.epsilon,
        )
        x = feature_frame[FEATURE_NAMES].to_numpy(dtype=float)
        predicted = float(model.estimator.predict(x)[0])
        if parameter.transform == "log1p":
            predicted = max(0.0, predicted)
        persistence = float(persistence_prediction(feature_frame)[0])
        linear = float(
            linear_extrapolation_prediction(
                feature_frame,
                baseline_h=self.config.baseline_h,
                early_h=self.config.early_h,
                target_h=self.config.target_h,
            )[0]
        )

        spec_min, spec_max, limit_codes = self._engineering_limits(series, parameter)
        guard_min, guard_max = self._guard_limits(spec_min, spec_max)
        reasons.extend(limit_codes)
        value_0 = float(base["value_0h"])
        value_24 = float(base["value_24h"])
        early_static_reasons = self._static_reasons(value_0, value_24, spec_min, spec_max)

        total_elapsed = self.config.target_h - self.config.baseline_h
        predicted_slope = (predicted - value_0) / total_elapsed
        danger_slope = float(direction_aware_slope(predicted_slope, parameter.danger_direction))
        conservative_value = self._conservative_endpoint(
            predicted,
            model.danger_residual_margin,
            parameter.danger_direction,
            value_0,
            value_24,
        )
        conservative_slope = (conservative_value - value_0) / total_elapsed
        conservative_danger_slope = float(
            direction_aware_slope(conservative_slope, parameter.danger_direction)
        )
        linear_slope = (linear - value_0) / total_elapsed
        linear_danger_slope = float(
            direction_aware_slope(linear_slope, parameter.danger_direction)
        )
        safety_slope, safety_sources, safety_codes = self._safety_slope(
            series,
            model,
            parameter.danger_direction,
            value_0,
            predicted,
            spec_min,
            spec_max,
        )
        reasons.extend(safety_codes)

        point_exceeds = safety_slope is not None and danger_slope > safety_slope
        interval_exceeds = (
            safety_slope is not None and conservative_danger_slope > safety_slope
        )
        linear_sentinel = (
            interval_exceeds
            and safety_slope is not None
            and linear_danger_slope
            > self.config.linear_sentinel_multiplier * safety_slope
        )
        if point_exceeds:
            reasons.append("PREDICTED_DRIFT_EXCEEDS_SAFETY_SLOPE")
        elif interval_exceeds:
            reasons.append("PREDICTION_INTERVAL_EXCEEDS_SAFETY_SLOPE")
        if linear_sentinel:
            reasons.append("EXTREME_LINEAR_BASELINE_CONFIRMS_UNCERTAIN_DRIFT")
        forecast_limit_reasons = self._forecast_limit_reasons(predicted, spec_min, spec_max)
        reasons.extend(forecast_limit_reasons)
        forecast_guard_reasons = self._forecast_guard_reasons(
            predicted, guard_min, guard_max, parameter.danger_direction
        )
        reasons.extend(forecast_guard_reasons)

        if early_static_reasons:
            decision = DriftDecision.STATIC_FAIL
            reasons.extend(early_static_reasons)
        elif safety_slope is None or limit_codes or safety_codes:
            decision = DriftDecision.RETEST_REQUIRED
        elif (
            point_exceeds
            or linear_sentinel
            or forecast_limit_reasons
            or forecast_guard_reasons
        ):
            decision = DriftDecision.EARLY_REJECT
        else:
            decision = DriftDecision.CONTINUE_SCREENING

        risk_denominator = max(safety_slope or 0.0, self.config.epsilon)
        model_risk = max(0.0, danger_slope) / risk_denominator
        sentinel_risk = (
            max(0.0, linear_danger_slope)
            / (self.config.linear_sentinel_multiplier * risk_denominator)
            if interval_exceeds
            else 0.0
        )
        risk_score = min(1.0, max(model_risk, sentinel_risk))
        if decision in {DriftDecision.STATIC_FAIL, DriftDecision.RETEST_REQUIRED}:
            risk_score = 1.0
        explanation = self._explanation(
            model,
            feature_frame,
            predicted,
            persistence,
            linear,
            danger_slope,
            conservative_danger_slope,
            safety_slope,
            safety_sources,
            parameter.canonical_unit,
        )
        return {
            **base,
            "predicted_value_168h": predicted,
            "persistence_prediction_168h": persistence,
            "linear_extrapolation_prediction_168h": linear,
            "predicted_total_drift": predicted - value_0,
            "predicted_drift_rate_per_h": predicted_slope,
            "danger_directed_drift_rate_per_h": danger_slope,
            "conservative_value_168h": conservative_value,
            "conservative_danger_drift_rate_per_h": conservative_danger_slope,
            "linear_extrapolation_danger_drift_rate_per_h": linear_danger_slope,
            "prediction_interval_exceeds_safety_slope": interval_exceeds,
            "safety_slope_per_h": safety_slope,
            "safety_slope_sources": safety_sources,
            "datasheet_min": spec_min,
            "datasheet_max": spec_max,
            "guard_min": guard_min,
            "guard_max": guard_max,
            "model": model.name,
            "model_scope": model_scope,
            "decision": decision.value,
            "flagged_for_early_rejection": decision
            in {DriftDecision.EARLY_REJECT, DriftDecision.STATIC_FAIL},
            "risk_score": risk_score,
            "reason_codes": list(dict.fromkeys(reasons)),
            "explanation": explanation,
        }

    def _unscored_result(self, base: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
        return {
            **base,
            "predicted_value_168h": None,
            "persistence_prediction_168h": None,
            "linear_extrapolation_prediction_168h": None,
            "predicted_total_drift": None,
            "predicted_drift_rate_per_h": None,
            "danger_directed_drift_rate_per_h": None,
            "conservative_value_168h": None,
            "conservative_danger_drift_rate_per_h": None,
            "linear_extrapolation_danger_drift_rate_per_h": None,
            "prediction_interval_exceeds_safety_slope": None,
            "safety_slope_per_h": None,
            "safety_slope_sources": [],
            "datasheet_min": None,
            "datasheet_max": None,
            "guard_min": None,
            "guard_max": None,
            "model": None,
            "model_scope": None,
            "decision": DriftDecision.RETEST_REQUIRED.value,
            "flagged_for_early_rejection": False,
            "risk_score": 1.0,
            "reason_codes": list(dict.fromkeys(reasons)),
            "explanation": {
                "summary": "A reliable forecast could not be produced; retest is required.",
                "feature_values": None,
                "contributions": None,
            },
        }

    def _engineering_limits(self, series, parameter):
        codes: list[str] = []
        source_unit = series.get("source_unit") or parameter.canonical_unit

        def converted(column: str, fallback: float | None) -> float | None:
            raw = _number(series.get(column))
            if raw is None:
                return fallback
            try:
                return convert_value(raw, str(source_unit), parameter.canonical_unit)
            except ValueError:
                codes.append(f"INVALID_{column.upper()}")
                return fallback

        return converted("datasheet_min", parameter.spec_min), converted(
            "datasheet_max", parameter.spec_max
        ), codes

    @staticmethod
    def _static_reasons(value_0, value_24, spec_min, spec_max) -> list[str]:
        reasons = []
        for label, value in [("0H", value_0), ("24H", value_24)]:
            if spec_min is not None and value < spec_min:
                reasons.append(f"STATIC_LIMIT_LOW_{label}")
            if spec_max is not None and value > spec_max:
                reasons.append(f"STATIC_LIMIT_HIGH_{label}")
        return reasons

    @staticmethod
    def _forecast_limit_reasons(predicted, spec_min, spec_max) -> list[str]:
        reasons = []
        if spec_min is not None and predicted < spec_min:
            reasons.append("PREDICTED_STATIC_LIMIT_LOW_168H")
        if spec_max is not None and predicted > spec_max:
            reasons.append("PREDICTED_STATIC_LIMIT_HIGH_168H")
        return reasons

    @staticmethod
    def _forecast_guard_reasons(predicted, guard_min, guard_max, direction) -> list[str]:
        reasons = []
        if direction in {DangerDirection.LOWER, DangerDirection.TWO_SIDED}:
            if guard_min is not None and predicted < guard_min:
                reasons.append("PREDICTED_GUARD_LIMIT_LOW_168H")
        if direction in {DangerDirection.HIGHER, DangerDirection.TWO_SIDED}:
            if guard_max is not None and predicted > guard_max:
                reasons.append("PREDICTED_GUARD_LIMIT_HIGH_168H")
        return reasons

    @staticmethod
    def _conservative_endpoint(predicted, margin, direction, value_0, value_24):
        if direction is DangerDirection.HIGHER:
            return predicted + margin
        if direction is DangerDirection.LOWER:
            return predicted - margin
        trend = predicted - value_0
        if abs(trend) <= 1e-15:
            trend = value_24 - value_0
        return predicted + (margin if trend >= 0 else -margin)

    def _safety_slope(
        self,
        series,
        model: FittedDriftModel,
        direction: DangerDirection,
        value_0: float,
        predicted: float,
        spec_min: float | None,
        spec_max: float | None,
    ) -> tuple[float | None, list[dict[str, Any]], list[str]]:
        candidates: list[tuple[str, float]] = [
            ("historical_good_part_quantile", model.historical_safety_slope)
        ]
        codes: list[str] = []
        source_unit = series.get("source_unit") or series.get("unit")
        delta_limit = _number(series.get("delta_limit"))
        if delta_limit is not None:
            if delta_limit < 0:
                codes.append("INVALID_DELTA_LIMIT")
            else:
                try:
                    canonical_delta = abs(
                        convert_value(delta_limit, str(source_unit), str(series.get("unit")))
                    )
                    candidates.append(
                        (
                            "configured_delta_limit",
                            canonical_delta / (self.config.target_h - self.config.baseline_h),
                        )
                    )
                except ValueError:
                    codes.append("INVALID_DELTA_LIMIT_UNIT")

        guard_min, guard_max = self._guard_limits(spec_min, spec_max)
        spec_slope: float | None = None
        elapsed = self.config.target_h - self.config.baseline_h
        if direction is DangerDirection.HIGHER and guard_max is not None:
            spec_slope = max(0.0, (guard_max - value_0) / elapsed)
        elif direction is DangerDirection.LOWER and guard_min is not None:
            spec_slope = max(0.0, (value_0 - guard_min) / elapsed)
        elif direction is DangerDirection.TWO_SIDED:
            moving_higher = predicted >= value_0
            if moving_higher and guard_max is not None:
                spec_slope = max(0.0, (guard_max - value_0) / elapsed)
            elif not moving_higher and guard_min is not None:
                spec_slope = max(0.0, (value_0 - guard_min) / elapsed)
        if spec_slope is not None:
            candidates.append(("datasheet_guard_band", spec_slope))

        finite = [(name, float(value)) for name, value in candidates if math.isfinite(value)]
        if not finite:
            return None, [], codes
        selected_name, selected_value = min(finite, key=lambda item: item[1])
        sources = [
            {
                "source": name,
                "slope_per_h": value,
                "binding": name == selected_name and value == selected_value,
            }
            for name, value in finite
        ]
        return max(0.0, selected_value), sources, codes

    def _guard_limits(
        self, spec_min: float | None, spec_max: float | None
    ) -> tuple[float | None, float | None]:
        if spec_min is not None and spec_max is not None:
            width = max(spec_max - spec_min, self.config.epsilon)
        elif spec_min is not None:
            width = max(abs(spec_min), self.config.epsilon)
        elif spec_max is not None:
            width = max(abs(spec_max), self.config.epsilon)
        else:
            return None, None
        guard = self.config.guard_band_fraction * width
        return (
            spec_min + guard if spec_min is not None else None,
            spec_max - guard if spec_max is not None else None,
        )

    @staticmethod
    def _explanation(
        model,
        features,
        predicted,
        persistence,
        linear,
        danger_slope,
        conservative_danger_slope,
        safety_slope,
        safety_sources,
        unit,
    ):
        feature_values = {
            name: float(features.iloc[0][name]) for name in model.feature_names
        }
        contributions = None
        estimator = model.estimator
        if hasattr(estimator, "named_steps"):
            scaler = estimator.named_steps.get("scale")
            regressor = estimator.named_steps.get("regressor")
            if scaler is not None and regressor is not None and hasattr(regressor, "coef_"):
                transformed = scaler.transform(features[model.feature_names].to_numpy(dtype=float))[0]
                coefficient = np.asarray(regressor.coef_, dtype=float).reshape(-1)
                contributions = {
                    "intercept": float(regressor.intercept_),
                    "terms": [
                        {
                            "feature": name,
                            "value": feature_values[name],
                            "standardized_value": float(transformed[index]),
                            "coefficient": float(coefficient[index]),
                            "contribution": float(coefficient[index] * transformed[index]),
                        }
                        for index, name in enumerate(model.feature_names)
                    ],
                }
                reconstructed = contributions["intercept"] + sum(
                    term["contribution"] for term in contributions["terms"]
                )
                contributions["postprocessing_adjustment"] = float(
                    predicted - reconstructed
                )
        threshold_text = (
            "unavailable"
            if safety_slope is None
            else f"{safety_slope:.6g} {unit}/h"
        )
        return {
            "summary": (
                f"{model.name} forecasts {predicted:.6g} {unit} at 168 h. "
                f"Danger-directed slope is {danger_slope:.6g} {unit}/h "
                f"(conservative bound {conservative_danger_slope:.6g}) versus "
                f"safety slope {threshold_text}."
            ),
            "model": model.name,
            "feature_values": feature_values,
            "contributions": contributions,
            "baselines": {
                "persistence": persistence,
                "linear_extrapolation": linear,
            },
            "calibration": {
                "training_samples": model.training_samples,
                "training_lots": model.training_lots,
                "cross_validated_mae": model.cv_mae,
                "persistence_mae": model.persistence_mae,
                "linear_extrapolation_mae": model.linear_extrapolation_mae,
                "improvement_over_persistence_percent": _relative_improvement(
                    model.cv_mae, model.persistence_mae
                ),
                "improvement_over_linear_percent": _relative_improvement(
                    model.cv_mae, model.linear_extrapolation_mae
                ),
                "danger_side_residual_margin": model.danger_residual_margin,
            },
            "safety_slope_sources": safety_sources,
        }

    def _aggregate_components(self, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in predictions:
            grouped[result["component_id"]].append(result)
        output = []
        for component_id, results in sorted(grouped.items()):
            decision = worst_decision(
                [DriftDecision(item["decision"]) for item in results]
            )
            highest = max(
                results,
                key=lambda item: (
                    DECISION_PRIORITY[DriftDecision(item["decision"])],
                    float(item.get("risk_score", 0.0)),
                ),
            )
            output.append(
                {
                    "component_id": component_id,
                    "decision": decision.value,
                    "flagged_for_early_rejection": any(
                        bool(item.get("flagged_for_early_rejection")) for item in results
                    ),
                    "risk_score": max(float(item.get("risk_score", 0.0)) for item in results),
                    "highest_risk_parameter": highest["parameter"],
                    "predicted_value_168h": highest.get("predicted_value_168h"),
                    "reason_codes": sorted(
                        {code for item in results for code in item.get("reason_codes", [])}
                    ),
                    "config_version": self.config.version,
                    "artifact_version": self.artifact.version if self.artifact else None,
                }
            )
        return output
