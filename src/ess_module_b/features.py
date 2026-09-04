"""Leakage-safe features derived exclusively from the 0-hour and 24-hour values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ess_module_a.models import DangerDirection


FEATURE_NAMES = [
    "value_0h",
    "value_24h",
    "delta_0_24",
    "early_slope",
    "relative_drift",
    "log_ratio",
]


def engineer_features(
    series: pd.DataFrame,
    *,
    baseline_h: float = 0.0,
    early_h: float = 24.0,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Create deterministic features; no value at or after 168 hours is consulted."""

    value_0 = pd.to_numeric(series["value_0h"], errors="coerce").astype(float)
    value_24 = pd.to_numeric(series["value_24h"], errors="coerce").astype(float)
    delta = value_24 - value_0
    elapsed = early_h - baseline_h
    denominator = np.maximum(np.abs(value_0.to_numpy(dtype=float)), epsilon)
    log_ratio = np.log(
        (np.abs(value_24.to_numpy(dtype=float)) + epsilon)
        / (np.abs(value_0.to_numpy(dtype=float)) + epsilon)
    )
    return pd.DataFrame(
        {
            "value_0h": value_0,
            "value_24h": value_24,
            "delta_0_24": delta,
            "early_slope": delta / elapsed,
            "relative_drift": delta.to_numpy(dtype=float) / denominator,
            "log_ratio": log_ratio,
        },
        index=series.index,
    )


def persistence_prediction(features: pd.DataFrame) -> np.ndarray:
    return features["value_24h"].to_numpy(dtype=float)


def linear_extrapolation_prediction(
    features: pd.DataFrame,
    *,
    baseline_h: float = 0.0,
    early_h: float = 24.0,
    target_h: float = 168.0,
) -> np.ndarray:
    remaining_intervals = (target_h - early_h) / (early_h - baseline_h)
    return (
        features["value_24h"].to_numpy(dtype=float)
        + remaining_intervals * features["delta_0_24"].to_numpy(dtype=float)
    )


def direction_aware_slope(slope: float | np.ndarray, direction: DangerDirection):
    values = np.asarray(slope, dtype=float)
    if direction is DangerDirection.HIGHER:
        output = values
    elif direction is DangerDirection.LOWER:
        output = -values
    else:
        output = np.abs(values)
    return float(output) if output.ndim == 0 else output
