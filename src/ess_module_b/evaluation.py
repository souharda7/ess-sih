"""Lot-safe accuracy and high-reliability screening evaluation for Module B."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ess_module_a.evaluation import split_lots

from .config import ModuleBConfig, default_config
from .data import SERIES_KEY, prepare_series
from .engine import ModuleBEngine


SAFE_FLAG_DECISIONS = {"EARLY_REJECT", "STATIC_FAIL", "RETEST_REQUIRED"}


def _series_key(record: dict[str, Any] | pd.Series) -> tuple[str, ...]:
    return tuple(str(record.get(column)) for column in SERIES_KEY)


def _regression_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "mae": None,
            "median_absolute_error": None,
            "rmse": None,
            "r2": None,
            "p95_absolute_error": None,
            "persistence_mae": None,
            "linear_extrapolation_mae": None,
            "improvement_over_persistence_percent": None,
            "improvement_over_linear_percent": None,
        }
    actual = np.asarray([record["actual"] for record in records], dtype=float)
    predicted = np.asarray([record["predicted"] for record in records], dtype=float)
    persistence = np.asarray([record["persistence"] for record in records], dtype=float)
    linear = np.asarray([record["linear"] for record in records], dtype=float)
    absolute_error = np.abs(actual - predicted)
    mae = float(mean_absolute_error(actual, predicted))
    persistence_mae = float(mean_absolute_error(actual, persistence))
    linear_mae = float(mean_absolute_error(actual, linear))

    def improvement(baseline: float) -> float | None:
        return None if baseline <= 0 else float(100.0 * (baseline - mae) / baseline)

    return {
        "count": len(records),
        "mae": mae,
        "median_absolute_error": float(np.median(absolute_error)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "p95_absolute_error": float(np.quantile(absolute_error, 0.95)),
        "persistence_mae": persistence_mae,
        "linear_extrapolation_mae": linear_mae,
        "improvement_over_persistence_percent": improvement(persistence_mae),
        "improvement_over_linear_percent": improvement(linear_mae),
    }


def evaluate_partition(
    engine: ModuleBEngine,
    measurements: pd.DataFrame,
) -> dict[str, Any]:
    prepared = prepare_series(
        measurements,
        engine.config,
        include_target=True,
        require_single_lot=False,
    )
    actual_lookup = {
        _series_key(row): row
        for _, row in prepared.series.loc[
            prepared.series["data_quality_status"].eq("VALID")
            & prepared.series["actual_value_168h"].notna()
        ].iterrows()
    }

    regression_records: list[dict[str, Any]] = []
    component_predictions: dict[str, bool] = defaultdict(bool)
    component_actual: dict[str, bool] = defaultdict(bool)
    component_lot: dict[str, str] = {}
    component_defect_types: dict[str, set[str]] = defaultdict(set)
    decisions: list[str] = []
    for _, actual_series in prepared.series.iterrows():
        component_id = str(actual_series["component_id"])
        is_anomaly = not bool(actual_series["is_good_reference"])
        component_actual[component_id] |= is_anomaly
        component_lot[component_id] = str(actual_series["lot_id"])
        if is_anomaly:
            defect_type = actual_series.get("defect_type")
            normalized_type = (
                str(defect_type)
                if defect_type is not None and str(defect_type).lower() != "nan"
                else "labelled_defect"
            )
            component_defect_types[component_id].add(normalized_type)

    for _, lot in measurements.groupby("lot_id", sort=False):
        report = engine.forecast_lot(lot)
        for component in report["component_results"]:
            component_predictions[component["component_id"]] |= (
                component["decision"] in SAFE_FLAG_DECISIONS
            )
            decisions.append(component["decision"])
        for prediction in report["prediction_results"]:
            actual_series = actual_lookup.get(_series_key(prediction))
            if actual_series is None or prediction["predicted_value_168h"] is None:
                continue
            regression_records.append(
                {
                    "parameter": prediction["parameter"],
                    "actual": float(actual_series["actual_value_168h"]),
                    "predicted": float(prediction["predicted_value_168h"]),
                    "persistence": float(prediction["persistence_prediction_168h"]),
                    "linear": float(prediction["linear_extrapolation_prediction_168h"]),
                }
            )

    component_ids = sorted(set(component_actual) | set(component_predictions))
    actual = np.asarray([component_actual[item] for item in component_ids], dtype=bool)
    predicted = np.asarray([component_predictions[item] for item in component_ids], dtype=bool)
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    recall = tp / (tp + fn) if tp + fn else None
    false_flag_rate = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    weighted_cost = engine.config.false_negative_cost * fn + fp
    false_negative_components = [
        component_id
        for component_id in component_ids
        if component_actual[component_id] and not component_predictions[component_id]
    ]

    recall_by_defect_type: dict[str, dict[str, int | float]] = {}
    all_defect_types = sorted(
        {item for values in component_defect_types.values() for item in values}
    )
    for defect_type in all_defect_types:
        affected = [
            component_id
            for component_id, values in component_defect_types.items()
            if defect_type in values
        ]
        detected = sum(component_predictions[component_id] for component_id in affected)
        recall_by_defect_type[defect_type] = {
            "count": len(affected),
            "detected": int(detected),
            "recall": float(detected / len(affected)),
        }

    lot_recalls: list[float] = []
    for lot_id in sorted(set(component_lot.values())):
        positives = [
            component_id
            for component_id in component_ids
            if component_lot.get(component_id) == lot_id and component_actual[component_id]
        ]
        if positives:
            lot_recalls.append(
                sum(component_predictions[component_id] for component_id in positives)
                / len(positives)
            )

    by_parameter: dict[str, Any] = {}
    for parameter, records in pd.DataFrame(regression_records).groupby("parameter") if regression_records else []:
        by_parameter[str(parameter)] = _regression_metrics(records.to_dict("records"))

    return {
        "target_h": engine.config.target_h,
        "regression": _regression_metrics(regression_records),
        "regression_by_parameter": by_parameter,
        "screening": {
            "component_count": len(component_ids),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "defect_recall": recall,
            "precision": precision,
            "false_flag_rate": false_flag_rate,
            "false_negative_cost_multiplier": engine.config.false_negative_cost,
            "weighted_error_cost": weighted_cost,
            "worst_lot_recall": min(lot_recalls) if lot_recalls else None,
            "recall_by_defect_type": recall_by_defect_type,
            "false_negative_components": false_negative_components,
            "decision_counts": pd.Series(decisions, dtype="string").value_counts().to_dict(),
        },
    }


def evaluate_lot_safe(
    measurements: pd.DataFrame,
    config: ModuleBConfig | None = None,
    *,
    seed: int = 170,
) -> dict[str, Any]:
    active_config = config or default_config()
    partitions = split_lots(measurements, seed=seed)
    engine = ModuleBEngine(active_config)
    engine.fit(partitions["train"])
    return {
        "split_lots": {
            name: sorted(str(value) for value in frame["lot_id"].unique())
            for name, frame in partitions.items()
        },
        "model_info": engine.model_info(),
        "validation": evaluate_partition(engine, partitions["validation"]),
        "test": evaluate_partition(engine, partitions["test"]),
    }
