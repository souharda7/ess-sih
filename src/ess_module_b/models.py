"""Serializable domain models for the 168-hour drift predictor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DriftDecision(str, Enum):
    """Module B's deliberately conservative early-screening outcomes."""

    CONTINUE_SCREENING = "CONTINUE_SCREENING"
    EARLY_REJECT = "EARLY_REJECT"
    STATIC_FAIL = "STATIC_FAIL"
    RETEST_REQUIRED = "RETEST_REQUIRED"


DECISION_PRIORITY = {
    DriftDecision.CONTINUE_SCREENING: 0,
    DriftDecision.EARLY_REJECT: 1,
    DriftDecision.RETEST_REQUIRED: 2,
    DriftDecision.STATIC_FAIL: 3,
}


@dataclass(slots=True)
class FittedDriftModel:
    """One fitted regression model and its out-of-fold calibration evidence."""

    name: str
    estimator: Any
    feature_names: list[str]
    training_samples: int
    training_lots: int
    cv_mae: float
    persistence_mae: float
    linear_extrapolation_mae: float
    candidate_mae: dict[str, float]
    danger_residual_margin: float
    historical_safety_slope: float
    target_min: float
    target_max: float


@dataclass(slots=True)
class DriftPredictorArtifact:
    """Versioned collection of exact-context and parameter fallback models."""

    version: str
    config_version: str
    created_at_utc: str
    training_lot_ids: list[str]
    context_models: dict[tuple[str, str, str], FittedDriftModel]
    parameter_models: dict[str, FittedDriftModel]
    library_versions: dict[str, str] = field(default_factory=dict)


def worst_decision(decisions: list[DriftDecision]) -> DriftDecision:
    if not decisions:
        return DriftDecision.CONTINUE_SCREENING
    return max(decisions, key=lambda decision: DECISION_PRIORITY[decision])
