"""Shared domain models for Module A."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DangerDirection(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"
    TWO_SIDED = "two_sided"


class QAStatus(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    QUARANTINE = "QUARANTINE"
    STATIC_FAIL = "STATIC_FAIL"
    RETEST_REQUIRED = "RETEST_REQUIRED"


STATUS_PRIORITY = {
    QAStatus.NORMAL: 0,
    QAStatus.MONITOR: 1,
    QAStatus.QUARANTINE: 2,
    QAStatus.RETEST_REQUIRED: 3,
    QAStatus.STATIC_FAIL: 4,
}


@dataclass(slots=True)
class DistributionStats:
    count: int
    median: float
    mad: float
    q1: float
    q3: float
    values: list[float] = field(repr=False)

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    def to_summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "median": self.median,
            "mad": self.mad,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
        }


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    component_id: str | None = None
    parameter: str | None = None
    time_h: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReferenceProfile:
    version: str
    config_version: str
    training_lot_ids: list[str]
    created_at_utc: str
    historical: dict[tuple[Any, ...], DistributionStats]
    historical_transformed: dict[tuple[Any, ...], DistributionStats]
    slopes: dict[tuple[Any, ...], DistributionStats]
    lot_medians: dict[tuple[Any, ...], DistributionStats]
    isolation_forest: Any | None = None
    isolation_training_scores: list[float] = field(default_factory=list)
    mahalanobis_models: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MahalanobisArtifact:
    parameters: list[str]
    estimator: Any
    training_scores: list[float]


@dataclass(slots=True)
class ValidationResult:
    measurements: Any
    issues: list[ValidationIssue]
    series_issues: dict[tuple[str, str, str], list[str]]
    lot_id: str | None
    as_of_h: float


def worst_status(statuses: list[QAStatus]) -> QAStatus:
    if not statuses:
        return QAStatus.NORMAL
    return max(statuses, key=lambda status: STATUS_PRIORITY[status])
