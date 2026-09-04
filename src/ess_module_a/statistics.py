"""Robust statistical primitives used by references and scoring."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from .models import DangerDirection, DistributionStats


def distribution_stats(values: Iterable[float]) -> DistributionStats:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot calculate distribution statistics without finite values")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    q1, q3 = np.percentile(array, [25, 75])
    return DistributionStats(
        count=int(array.size),
        median=median,
        mad=mad,
        q1=float(q1),
        q3=float(q3),
        values=np.sort(array).astype(float).tolist(),
    )


def robust_z(value: float, stats: DistributionStats, epsilon: float = 1e-9) -> float:
    scale = max(1.4826 * stats.mad, epsilon)
    return float((value - stats.median) / scale)


def direction_risk(value: float, direction: DangerDirection) -> float:
    if not math.isfinite(value):
        return float("nan")
    if direction is DangerDirection.HIGHER:
        return float(value)
    if direction is DangerDirection.LOWER:
        return float(-value)
    return float(abs(value))


def empirical_percentile(value: float, stats: DistributionStats) -> float:
    values = np.asarray(stats.values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(100.0 * np.searchsorted(values, value, side="right") / values.size)


def tail_percentile(percentile: float, direction: DangerDirection) -> float:
    if direction is DangerDirection.HIGHER:
        return percentile
    if direction is DangerDirection.LOWER:
        return 100.0 - percentile
    return max(percentile, 100.0 - percentile)


def iqr_distance(value: float, stats: DistributionStats, epsilon: float = 1e-9) -> float:
    """Return signed IQR units beyond the nearest quartile; zero within Q1-Q3."""
    iqr = max(stats.iqr, epsilon)
    if value > stats.q3:
        return float((value - stats.q3) / iqr)
    if value < stats.q1:
        return float((value - stats.q1) / iqr)
    return 0.0


def transformed_value(value: float, transform: str) -> float:
    if transform == "raw":
        return float(value)
    if transform == "log1p":
        if value < 0:
            raise ValueError("log1p transform requires a non-negative value")
        return float(np.log1p(value))
    raise ValueError(f"Unsupported transform: {transform}")


def percentile_of_scores(score: float, reference_scores: list[float]) -> float | None:
    if not reference_scores or not math.isfinite(score):
        return None
    values = np.sort(np.asarray(reference_scores, dtype=float))
    return float(100.0 * np.searchsorted(values, score, side="right") / values.size)

