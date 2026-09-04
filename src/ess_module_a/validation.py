"""Batch validation and normalization for long-form ESS measurements."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import pandas as pd

from .config import ModuleAConfig
from .models import ValidationIssue, ValidationResult
from .statistics import transformed_value
from .units import convert_value


REQUIRED_COLUMNS = {
    "component_id",
    "lot_id",
    "part_number",
    "parameter",
    "time_h",
    "value",
    "unit",
    "test_condition_id",
}

DUPLICATE_KEY = [
    "component_id",
    "lot_id",
    "part_number",
    "parameter",
    "time_h",
    "test_condition_id",
]


class DataValidationError(ValueError):
    """Raised when a batch cannot be interpreted at all."""


def _series_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row["component_id"]), str(row["parameter"]), str(row["test_condition_id"]))


def validate_measurements(
    measurements: pd.DataFrame,
    config: ModuleAConfig,
    *,
    as_of_h: float | None = None,
    require_single_lot: bool = True,
) -> ValidationResult:
    missing_columns = sorted(REQUIRED_COLUMNS - set(measurements.columns))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")
    if measurements.empty:
        raise DataValidationError("Measurement batch is empty")

    frame = measurements.copy()
    frame["_valid"] = True
    frame["_issues"] = [[] for _ in range(len(frame))]
    frame["normalized_value"] = float("nan")
    frame["transformed_value"] = float("nan")

    for column in ["component_id", "lot_id", "part_number", "parameter", "unit", "test_condition_id"]:
        frame[column] = frame[column].astype("string")

    frame["time_h"] = pd.to_numeric(frame["time_h"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    lots = sorted(str(x) for x in frame["lot_id"].dropna().unique())
    if require_single_lot and len(lots) != 1:
        raise DataValidationError("score_lot requires measurements from exactly one lot")

    available_times = frame["time_h"].dropna()
    effective_as_of = float(as_of_h if as_of_h is not None else available_times.max())
    if effective_as_of not in config.checkpoints_h:
        raise DataValidationError(
            f"as_of_h={effective_as_of:g} is not in configured checkpoints {config.checkpoints_h}"
        )

    issues: list[ValidationIssue] = []
    series_issues: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    def mark(index: Any, code: str, message: str, *, critical_series: bool = True) -> None:
        row = frame.loc[index]
        frame.at[index, "_valid"] = False
        frame.at[index, "_issues"] = [*frame.at[index, "_issues"], code]
        issue = ValidationIssue(
            code=code,
            message=message,
            component_id=None if pd.isna(row["component_id"]) else str(row["component_id"]),
            parameter=None if pd.isna(row["parameter"]) else str(row["parameter"]),
            time_h=None if pd.isna(row["time_h"]) else float(row["time_h"]),
        )
        issues.append(issue)
        if critical_series and all(
            not pd.isna(row[column])
            for column in ("component_id", "parameter", "test_condition_id")
        ):
            series_issues[_series_key(row)].append(code)

    for index, row in frame.iterrows():
        for column in ["component_id", "lot_id", "part_number", "parameter", "unit", "test_condition_id"]:
            if pd.isna(row[column]) or not str(row[column]).strip():
                mark(index, "MISSING_REQUIRED_VALUE", f"{column} is required")
                break

        if pd.isna(row["time_h"]) or not math.isfinite(float(row["time_h"])):
            mark(index, "INVALID_CHECKPOINT", "time_h must be finite and numeric")
            continue
        if float(row["time_h"]) not in config.checkpoints_h:
            mark(index, "INVALID_CHECKPOINT", f"Unsupported checkpoint {row['time_h']}")
        if float(row["time_h"]) > effective_as_of:
            mark(
                index,
                "FUTURE_MEASUREMENT_IGNORED",
                "Measurement occurs after as_of_h and was excluded from scoring",
                critical_series=False,
            )

        parameter_name = str(row["parameter"])
        parameter = config.parameters.get(parameter_name)
        if parameter is None:
            mark(index, "UNKNOWN_PARAMETER", f"No configuration for {parameter_name}")
            continue

        for context_field in parameter.required_context_fields:
            if context_field not in frame.columns or pd.isna(row.get(context_field)) or str(row.get(context_field)).strip() == "":
                mark(index, "MISSING_TEST_CONTEXT", f"{context_field} is required for {parameter_name}")

        if pd.isna(row["value"]) or not math.isfinite(float(row["value"])):
            mark(index, "INVALID_VALUE", "value must be finite and numeric")
            continue
        try:
            normalized = convert_value(float(row["value"]), str(row["unit"]), parameter.canonical_unit)
            transformed = transformed_value(normalized, parameter.transform)
            frame.at[index, "normalized_value"] = normalized
            frame.at[index, "transformed_value"] = transformed
        except ValueError as exc:
            mark(index, "UNIT_OR_TRANSFORM_ERROR", str(exc))

    duplicate_mask = frame.duplicated(DUPLICATE_KEY, keep=False)
    for index in frame.index[duplicate_mask]:
        mark(index, "DUPLICATE_MEASUREMENT", "Duplicate component/parameter/checkpoint/condition")

    expected_times = {time for time in config.checkpoints_h if time <= effective_as_of}
    group_columns = ["component_id", "parameter", "test_condition_id"]
    for key, group in frame.groupby(group_columns, dropna=False):
        if any(pd.isna(value) for value in key):
            continue
        present = set(float(x) for x in group.loc[group["_valid"], "time_h"].dropna())
        missing = sorted(expected_times - present)
        if missing:
            normalized_key = tuple(str(value) for value in key)
            code = "MISSING_CHECKPOINT"
            series_issues[normalized_key].append(code)
            issues.append(
                ValidationIssue(
                    code=code,
                    message=f"Missing required checkpoints through {effective_as_of:g} h: {missing}",
                    component_id=normalized_key[0],
                    parameter=normalized_key[1],
                )
            )

    return ValidationResult(
        measurements=frame,
        issues=issues,
        series_issues=dict(series_issues),
        lot_id=lots[0] if len(lots) == 1 else None,
        as_of_h=effective_as_of,
    )
