"""Input adaptation and series construction for Module B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from ess_module_a.validation import DataValidationError, validate_measurements

from .config import ModuleBConfig


SERIES_KEY = [
    "component_id",
    "lot_id",
    "part_number",
    "parameter",
    "test_condition_id",
]

_ALIASES = {
    "parameter_name": "parameter",
    "temperature_C": "temperature_c",
    "bias_voltage": "voltage_v",
}

_METADATA_COLUMNS = [
    "unit",
    "temperature_c",
    "voltage_v",
    "test_mode",
    "tester_id",
    "chamber_id",
    "socket_id",
    "datasheet_min",
    "datasheet_max",
    "delta_limit",
    "actual_value_168h",
    "is_anomaly",
    "defect_label",
    "qa_approved",
    "qa_disposition",
    "defect_type",
]


@dataclass(slots=True)
class PreparedSeries:
    series: pd.DataFrame
    validation_issues: list[dict[str, Any]]
    ignored_measurement_count: int


def _find_column(frame: pd.DataFrame, *names: str) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def _normalize_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename: dict[str, str] = {}
    for source, target in _ALIASES.items():
        if target not in result.columns and source in result.columns:
            rename[source] = target
    return result.rename(columns=rename)


def coerce_long_measurements(
    measurements: pd.DataFrame,
    config: ModuleBConfig,
    *,
    include_target: bool,
) -> pd.DataFrame:
    """Accept Module A long form or organizer-style Value_0h/Value_24h rows."""

    frame = _normalize_aliases(measurements)
    if {"time_h", "value"}.issubset(frame.columns):
        result = frame.copy()
    else:
        value_0_column = _find_column(frame, "Value_0h", "value_0h")
        value_24_column = _find_column(frame, "Value_24h", "value_24h")
        if value_0_column is None or value_24_column is None:
            raise DataValidationError(
                "Expected long-form time_h/value columns or wide Value_0h/Value_24h columns"
            )
        checkpoints: list[tuple[str, float]] = [
            (value_0_column, config.baseline_h),
            (value_24_column, config.early_h),
        ]
        target_column = _find_column(
            frame, "Value_168h", "value_168h", "actual_value_168h"
        )
        if include_target:
            if target_column is not None:
                checkpoints.append((target_column, config.target_h))
        checkpoint_columns = {column for column, _ in checkpoints}
        if target_column is not None:
            checkpoint_columns.add(target_column)
        base_columns = [column for column in frame.columns if column not in checkpoint_columns]
        parts = []
        for column, time_h in checkpoints:
            part = frame[base_columns].copy()
            part["time_h"] = float(time_h)
            part["value"] = frame[column]
            parts.append(part)
        result = pd.concat(parts, ignore_index=True)

    if include_target:
        result = _append_actual_targets(result, config)
    else:
        # Remove hidden labels before any validation or feature-engineering path can see them.
        hidden_columns = [
            column
            for column in result.columns
            if str(column).lower() in {"value_168h", "actual_value_168h"}
        ]
        result = result.drop(columns=hidden_columns)
    return result


def _append_actual_targets(frame: pd.DataFrame, config: ModuleBConfig) -> pd.DataFrame:
    """Materialize actual_value_168h when the target is supplied beside early rows."""

    if "actual_value_168h" not in frame.columns or not set(SERIES_KEY).issubset(frame.columns):
        return frame
    result = frame.copy()
    numeric_time = pd.to_numeric(result.get("time_h"), errors="coerce")
    existing = {
        tuple(str(value) for value in key)
        for key in result.loc[numeric_time == config.target_h, SERIES_KEY].itertuples(
            index=False, name=None
        )
    }
    additions: list[pd.Series] = []
    for key, group in result.groupby(SERIES_KEY, sort=False, dropna=False):
        normalized_key = tuple(str(value) for value in key)
        if normalized_key in existing:
            continue
        targets = pd.to_numeric(group["actual_value_168h"], errors="coerce").dropna()
        if targets.empty:
            continue
        finite_targets = targets[np.isfinite(targets.to_numpy(dtype=float))]
        if finite_targets.empty:
            continue
        first = float(finite_targets.iloc[0])
        if not np.allclose(finite_targets.to_numpy(dtype=float), first, rtol=1e-9, atol=1e-12):
            raise DataValidationError(
                f"Conflicting actual_value_168h values for series {normalized_key}"
            )
        preferred = group.loc[pd.to_numeric(group["time_h"], errors="coerce") == config.early_h]
        source = (preferred if not preferred.empty else group).iloc[0].copy()
        source["time_h"] = config.target_h
        source["value"] = first
        additions.append(source)
    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True)
    return result


def _first_value(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return None
    values = group[column].dropna()
    if values.empty:
        return None
    value = values.iloc[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _bool_value(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "defect", "defective", "fail", "failed"}:
        return True
    if normalized in {"0", "false", "no", "n", "normal", "good", "pass", "passed"}:
        return False
    return None


def is_good_reference(record: pd.Series | dict[str, Any]) -> bool:
    """Identify rows suitable for calculating a normal-population safety slope."""

    approved = _bool_value(record.get("qa_approved"))
    if approved is not None:
        return approved
    anomaly = _bool_value(record.get("is_anomaly"))
    if anomaly is not None:
        return not anomaly
    defect = _bool_value(record.get("defect_label"))
    if defect is not None:
        return not defect
    disposition = record.get("qa_disposition")
    if disposition is not None:
        return str(disposition).strip().lower() in {
            "accept",
            "accepted",
            "approve",
            "approved",
            "good",
            "normal",
            "pass",
            "passed",
        }
    return True


def prepare_series(
    measurements: pd.DataFrame,
    config: ModuleBConfig,
    *,
    include_target: bool,
    require_single_lot: bool,
) -> PreparedSeries:
    required_times = (
        (config.baseline_h, config.early_h, config.target_h)
        if include_target
        else (config.baseline_h, config.early_h)
    )
    long_frame = coerce_long_measurements(measurements, config, include_target=include_target)
    if "time_h" not in long_frame.columns:
        raise DataValidationError("time_h is required")
    numeric_time = pd.to_numeric(long_frame["time_h"], errors="coerce")
    relevant_mask = numeric_time.isna() | numeric_time.isin(required_times)
    ignored_count = int((~relevant_mask).sum())
    relevant = long_frame.loc[relevant_mask].copy()
    validation = validate_measurements(
        relevant,
        config.validation_config(tuple(float(x) for x in required_times)),
        as_of_h=float(required_times[-1]),
        require_single_lot=require_single_lot,
    )
    frame = validation.measurements
    records: list[dict[str, Any]] = []
    if set(SERIES_KEY).issubset(frame.columns):
        for key, group in frame.groupby(SERIES_KEY, sort=False, dropna=False):
            if any(pd.isna(value) for value in key):
                continue
            component_id, lot_id, part_number, parameter, condition = (
                str(value) for value in key
            )
            codes = [
                code
                for values in group.get("_issues", pd.Series([], dtype=object))
                for code in (values or [])
            ]
            valid = group.loc[group["_valid"]].copy()
            values_by_time = {
                float(row["time_h"]): float(row["normalized_value"])
                for _, row in valid.iterrows()
                if pd.notna(row["time_h"]) and pd.notna(row["normalized_value"])
            }
            if any(float(time_h) not in values_by_time for time_h in required_times):
                codes.append("MISSING_CHECKPOINT")
            for engineering_column in ("datasheet_min", "datasheet_max", "delta_limit"):
                if engineering_column not in group.columns:
                    continue
                engineering_values = pd.to_numeric(
                    group[engineering_column], errors="coerce"
                ).dropna()
                if len(engineering_values) > 1 and not np.allclose(
                    engineering_values.to_numpy(dtype=float),
                    float(engineering_values.iloc[0]),
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    codes.append(f"CONFLICTING_{engineering_column.upper()}")
            record: dict[str, Any] = {
                "component_id": component_id,
                "lot_id": lot_id,
                "part_number": part_number,
                "parameter": parameter,
                "test_condition_id": condition,
                "value_0h": values_by_time.get(config.baseline_h),
                "value_24h": values_by_time.get(config.early_h),
                "data_quality_status": "VALID" if not codes else "INVALID",
                "reason_codes": list(dict.fromkeys(codes)),
            }
            if include_target:
                record["actual_value_168h"] = values_by_time.get(config.target_h)
            for column in _METADATA_COLUMNS:
                if column == "unit":
                    parameter_config = config.parameters.get(parameter)
                    record["unit"] = (
                        parameter_config.canonical_unit
                        if parameter_config is not None
                        else _first_value(group, column)
                    )
                    record["source_unit"] = _first_value(group, column)
                elif column != "actual_value_168h":
                    record[column] = _first_value(group, column)
            record["is_good_reference"] = is_good_reference(record)
            records.append(record)

    issues = [issue.to_dict() for issue in validation.issues]
    if ignored_count:
        issues.append(
            {
                "code": "UNUSED_CHECKPOINT_IGNORED",
                "message": (
                    f"Ignored {ignored_count} measurement(s) outside Module B checkpoints "
                    f"{list(required_times)}"
                ),
                "component_id": None,
                "parameter": None,
                "time_h": None,
            }
        )
    return PreparedSeries(
        series=pd.DataFrame.from_records(records),
        validation_issues=issues,
        ignored_measurement_count=ignored_count,
    )
