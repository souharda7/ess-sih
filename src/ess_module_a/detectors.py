"""Direction-aware static and robust rule evaluation."""

from __future__ import annotations

import math
from typing import Any

from .config import ModuleAConfig, ParameterConfig
from .models import DangerDirection
from .statistics import direction_risk


def _is_number(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def static_check(value: float, parameter: ParameterConfig) -> tuple[bool, float | None, list[str]]:
    failed = False
    reasons: list[str] = []
    margins: list[float] = []
    if parameter.spec_min is not None:
        margins.append(value - parameter.spec_min)
        if value < parameter.spec_min:
            failed = True
            reasons.append("STATIC_LIMIT_LOW")
    if parameter.spec_max is not None:
        margins.append(parameter.spec_max - value)
        if value > parameter.spec_max:
            failed = True
            reasons.append("STATIC_LIMIT_HIGH")
    margin = min(margins) if margins else None
    return failed, margin, reasons


def evaluate_robust_rules(
    row: dict[str, Any],
    parameter: ParameterConfig,
    config: ModuleAConfig,
) -> dict[str, Any]:
    warnings: set[str] = set()
    severe: set[str] = set()
    reasons: list[str] = []
    extreme = False

    def z_rule(field: str, label: str, category: str) -> None:
        nonlocal extreme
        value = row.get(field)
        if not _is_number(value):
            return
        risk = direction_risk(float(value), parameter.danger_direction)
        if risk >= config.robust_z_extreme:
            extreme = True
            severe.add(category)
            reasons.append(f"{label}_EXTREME")
        elif risk >= config.robust_z_severe:
            severe.add(category)
            reasons.append(f"{label}_SEVERE")
        elif risk >= config.robust_z_warning:
            warnings.add(category)
            reasons.append(f"{label}_WARNING")

    z_rule("lot_z", "LOT_ROBUST_Z", "level_lot")
    z_rule("historical_z", "HISTORICAL_ROBUST_Z", "level_historical")
    z_rule("slope_lot_z", "LOT_SLOPE_Z", "slope_lot")
    z_rule("slope_historical_z", "HISTORICAL_SLOPE_Z", "slope_historical")

    for field, label, category in [
        ("lot_iqr_distance", "LOT_IQR", "level_lot"),
        ("historical_iqr_distance", "HISTORICAL_IQR", "level_historical"),
    ]:
        value = row.get(field)
        if not _is_number(value):
            continue
        risk = direction_risk(float(value), parameter.danger_direction)
        if risk >= config.iqr_severe_multiplier:
            severe.add(category)
            reasons.append(f"{label}_OUTER_FENCE")
        elif risk >= config.iqr_warning_multiplier:
            warnings.add(category)
            reasons.append(f"{label}_INNER_FENCE")

    for field, label, category in [
        ("lot_tail_percentile", "LOT_PERCENTILE", "level_lot"),
        ("historical_tail_percentile", "HISTORICAL_PERCENTILE", "level_historical"),
    ]:
        value = row.get(field)
        if not _is_number(value):
            continue
        if float(value) >= config.tail_severe_percentile:
            severe.add(category)
            reasons.append(f"{label}_SEVERE")
        elif float(value) >= config.tail_warning_percentile:
            warnings.add(category)
            reasons.append(f"{label}_WARNING")

    return {
        "warning_categories": warnings,
        "severe_categories": severe,
        "extreme": extreme,
        "reason_codes": list(dict.fromkeys(reasons)),
    }
