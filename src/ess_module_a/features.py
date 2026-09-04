"""Interpretable value, slope, and lot-shift feature engineering."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import numpy as np
import pandas as pd

from .config import ModuleAConfig
from .models import DistributionStats, ReferenceProfile
from .statistics import (
    direction_risk,
    distribution_stats,
    empirical_percentile,
    iqr_distance,
    robust_z,
    tail_percentile,
)


IFOREST_FEATURES = [
    "lot_z_signed",
    "historical_z_signed",
    "lot_z_abs",
    "historical_z_abs",
    "lot_iqr_abs",
    "historical_tail_extremeness",
    "slope_lot_z_signed",
    "slope_historical_z_signed",
    "slope_z_abs",
    "lot_shift_z_signed",
]


def history_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        str(row["part_number"]),
        str(row["parameter"]),
        float(row["time_h"]),
        str(row["test_condition_id"]),
    )


def lot_key(row: pd.Series) -> tuple[Any, ...]:
    return (str(row["lot_id"]), *history_key(row))


def slope_history_key(
    part_number: str,
    parameter: str,
    start_h: float,
    end_h: float,
    condition: str,
) -> tuple[Any, ...]:
    return (str(part_number), str(parameter), float(start_h), float(end_h), str(condition))


def compute_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_index",
        "component_id",
        "lot_id",
        "part_number",
        "parameter",
        "test_condition_id",
        "start_h",
        "end_h",
        "slope",
        "slope_from_zero",
    ]
    records: list[dict[str, Any]] = []
    group_columns = ["component_id", "lot_id", "part_number", "parameter", "test_condition_id"]
    for _, group in frame.loc[frame["_valid"]].groupby(group_columns, sort=False):
        ordered = group.sort_values("time_h")
        previous: pd.Series | None = None
        first: pd.Series | None = None
        for index, row in ordered.iterrows():
            if first is None:
                first = row
            if previous is None:
                previous = row
                continue
            elapsed = float(row["time_h"] - previous["time_h"])
            total_elapsed = float(row["time_h"] - first["time_h"])
            if elapsed <= 0 or total_elapsed <= 0:
                previous = row
                continue
            records.append(
                {
                    "row_index": index,
                    "component_id": str(row["component_id"]),
                    "lot_id": str(row["lot_id"]),
                    "part_number": str(row["part_number"]),
                    "parameter": str(row["parameter"]),
                    "test_condition_id": str(row["test_condition_id"]),
                    "start_h": float(previous["time_h"]),
                    "end_h": float(row["time_h"]),
                    "slope": float((row["normalized_value"] - previous["normalized_value"]) / elapsed),
                    "slope_from_zero": float(
                        (row["normalized_value"] - first["normalized_value"]) / total_elapsed
                    ),
                }
            )
            previous = row
    return pd.DataFrame.from_records(records, columns=columns)


def grouped_stats(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> dict[tuple[Any, ...], DistributionStats]:
    output: dict[tuple[Any, ...], DistributionStats] = {}
    valid = frame.loc[frame["_valid"] & frame[value_column].notna()]
    for key, group in valid.groupby(group_columns, sort=False):
        normalized_key = key if isinstance(key, tuple) else (key,)
        output[tuple(normalized_key)] = distribution_stats(group[value_column].astype(float))
    return output


def _available(stats: DistributionStats | None, minimum: int) -> DistributionStats | None:
    return stats if stats is not None and stats.count >= minimum else None


def _safe_number(value: float | None, default: float = 0.0) -> float:
    if value is None or not math.isfinite(value):
        return default
    return float(value)


def build_feature_frame(
    frame: pd.DataFrame,
    config: ModuleAConfig,
    reference: ReferenceProfile,
) -> pd.DataFrame:
    valid = frame.loc[frame["_valid"]].copy()
    lot_group_columns = [
        "lot_id",
        "part_number",
        "parameter",
        "time_h",
        "test_condition_id",
    ]
    lot_raw = grouped_stats(valid, lot_group_columns, "normalized_value")
    lot_transformed = grouped_stats(valid, lot_group_columns, "transformed_value")
    slopes = compute_slopes(frame)
    slope_by_row = {record["row_index"]: record for record in slopes.to_dict("records")}
    slope_lot: dict[tuple[Any, ...], DistributionStats] = {}
    if not slopes.empty:
        group_columns = [
            "lot_id",
            "part_number",
            "parameter",
            "start_h",
            "end_h",
            "test_condition_id",
        ]
        for key, group in slopes.groupby(group_columns, sort=False):
            slope_lot[tuple(key)] = distribution_stats(group["slope"])

    records: list[dict[str, Any]] = []
    for index, row in valid.iterrows():
        parameter = config.parameters[str(row["parameter"])]
        h_key = history_key(row)
        l_key = lot_key(row)
        current_raw = _available(lot_raw.get(l_key), parameter.minimum_lot_peers)
        current_transformed = _available(lot_transformed.get(l_key), parameter.minimum_lot_peers)
        historical_raw = _available(reference.historical.get(h_key), parameter.minimum_historical_peers)
        historical_transformed = _available(
            reference.historical_transformed.get(h_key), parameter.minimum_historical_peers
        )

        value = float(row["normalized_value"])
        transformed = float(row["transformed_value"])
        lot_z = robust_z(transformed, current_transformed, config.epsilon) if current_transformed else None
        historical_z = (
            robust_z(transformed, historical_transformed, config.epsilon)
            if historical_transformed
            else None
        )
        lot_percentile = empirical_percentile(value, current_raw) if current_raw else None
        historical_percentile = empirical_percentile(value, historical_raw) if historical_raw else None
        lot_iqr = iqr_distance(value, current_raw, config.epsilon) if current_raw else None
        historical_iqr = iqr_distance(value, historical_raw, config.epsilon) if historical_raw else None
        percentage_from_lot = None
        if current_raw:
            denominator = max(abs(current_raw.median), config.epsilon)
            percentage_from_lot = 100.0 * (value - current_raw.median) / denominator

        lot_shift_z = None
        lot_median_reference = reference.lot_medians.get(h_key)
        if current_raw and lot_median_reference and lot_median_reference.count >= 3:
            lot_shift_z = robust_z(current_raw.median, lot_median_reference, config.epsilon)

        slope_record = slope_by_row.get(index)
        slope = None
        slope_from_zero = None
        slope_lot_z = None
        slope_historical_z = None
        slope_start_h = None
        if slope_record:
            slope = float(slope_record["slope"])
            slope_from_zero = float(slope_record["slope_from_zero"])
            slope_start_h = float(slope_record["start_h"])
            lot_slope_key = (
                str(row["lot_id"]),
                str(row["part_number"]),
                str(row["parameter"]),
                slope_start_h,
                float(row["time_h"]),
                str(row["test_condition_id"]),
            )
            hist_slope_key = slope_history_key(
                str(row["part_number"]),
                str(row["parameter"]),
                slope_start_h,
                float(row["time_h"]),
                str(row["test_condition_id"]),
            )
            current_slope_stats = _available(
                slope_lot.get(lot_slope_key), parameter.minimum_lot_peers
            )
            historical_slope_stats = _available(
                reference.slopes.get(hist_slope_key), parameter.minimum_historical_peers
            )
            if current_slope_stats:
                slope_lot_z = robust_z(slope, current_slope_stats, config.epsilon)
            if historical_slope_stats:
                slope_historical_z = robust_z(slope, historical_slope_stats, config.epsilon)

        lot_tail = (
            tail_percentile(lot_percentile, parameter.danger_direction)
            if lot_percentile is not None
            else None
        )
        history_tail = (
            tail_percentile(historical_percentile, parameter.danger_direction)
            if historical_percentile is not None
            else None
        )
        risk_lot_z = direction_risk(lot_z, parameter.danger_direction) if lot_z is not None else None
        risk_history_z = (
            direction_risk(historical_z, parameter.danger_direction)
            if historical_z is not None
            else None
        )
        risk_slope_lot_z = (
            direction_risk(slope_lot_z, parameter.danger_direction)
            if slope_lot_z is not None
            else None
        )
        risk_slope_history_z = (
            direction_risk(slope_historical_z, parameter.danger_direction)
            if slope_historical_z is not None
            else None
        )
        risk_lot_shift_z = (
            direction_risk(lot_shift_z, parameter.danger_direction)
            if lot_shift_z is not None
            else None
        )

        record = row.to_dict()
        record.update(
            {
                "row_index": index,
                "lot_stats": current_raw,
                "historical_stats": historical_raw,
                "lot_z": lot_z,
                "historical_z": historical_z,
                "risk_lot_z": risk_lot_z,
                "risk_historical_z": risk_history_z,
                "lot_percentile": lot_percentile,
                "historical_percentile": historical_percentile,
                "lot_tail_percentile": lot_tail,
                "historical_tail_percentile": history_tail,
                "lot_iqr_distance": lot_iqr,
                "historical_iqr_distance": historical_iqr,
                "percentage_from_lot_median": percentage_from_lot,
                "lot_shift_z": lot_shift_z,
                "risk_lot_shift_z": risk_lot_shift_z,
                "slope": slope,
                "slope_from_zero": slope_from_zero,
                "slope_start_h": slope_start_h,
                "slope_lot_z": slope_lot_z,
                "slope_historical_z": slope_historical_z,
                "risk_slope_lot_z": risk_slope_lot_z,
                "risk_slope_historical_z": risk_slope_history_z,
                "lot_z_signed": _safe_number(lot_z),
                "historical_z_signed": _safe_number(historical_z),
                "lot_z_abs": abs(_safe_number(lot_z)),
                "historical_z_abs": abs(_safe_number(historical_z)),
                "lot_iqr_abs": abs(_safe_number(lot_iqr)),
                "historical_tail_extremeness": abs(_safe_number(history_tail, 50.0) - 50.0) / 50.0,
                "slope_lot_z_signed": _safe_number(slope_lot_z),
                "slope_historical_z_signed": _safe_number(slope_historical_z),
                "slope_z_abs": max(
                    abs(_safe_number(slope_lot_z)), abs(_safe_number(slope_historical_z))
                ),
                "lot_shift_z_signed": _safe_number(lot_shift_z),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def feature_matrix(feature_frame: pd.DataFrame) -> np.ndarray:
    if feature_frame.empty:
        return np.empty((0, len(IFOREST_FEATURES)), dtype=float)
    return (
        feature_frame.reindex(columns=IFOREST_FEATURES)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
