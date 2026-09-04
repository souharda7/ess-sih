"""Historical reference fitting and persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.covariance import MinCovDet

from .config import ModuleAConfig
from .features import build_feature_frame, compute_slopes
from .models import MahalanobisArtifact, ReferenceProfile
from .statistics import distribution_stats
from .validation import validate_measurements


HISTORY_GROUP = ["part_number", "parameter", "time_h", "test_condition_id"]
SLOPE_GROUP = [
    "part_number",
    "parameter",
    "start_h",
    "end_h",
    "test_condition_id",
]


def _stats_by_group(
    frame: pd.DataFrame, columns: list[str], value_column: str
) -> dict[tuple[Any, ...], Any]:
    output = {}
    for key, group in frame.groupby(columns, sort=False):
        normalized = key if isinstance(key, tuple) else (key,)
        output[tuple(normalized)] = distribution_stats(group[value_column].astype(float))
    return output


def fit_reference_profile(measurements: pd.DataFrame, config: ModuleAConfig) -> ReferenceProfile:
    as_of_h = max(config.checkpoints_h)
    validation = validate_measurements(
        measurements, config, as_of_h=as_of_h, require_single_lot=False
    )
    frame = validation.measurements.loc[validation.measurements["_valid"]].copy()
    if "qa_approved" in frame.columns:
        approved = frame["qa_approved"].fillna(False).astype(bool)
        if approved.any():
            frame = frame.loc[approved].copy()
    if frame.empty:
        raise ValueError("No valid measurements are available for reference fitting")

    historical = _stats_by_group(frame, HISTORY_GROUP, "normalized_value")
    historical_transformed = _stats_by_group(frame, HISTORY_GROUP, "transformed_value")
    slopes = compute_slopes(frame)
    slope_stats = _stats_by_group(slopes, SLOPE_GROUP, "slope") if not slopes.empty else {}

    lot_median_rows = (
        frame.groupby(["lot_id", *HISTORY_GROUP], as_index=False)["normalized_value"]
        .median()
        .rename(columns={"normalized_value": "lot_median"})
    )
    lot_medians = _stats_by_group(lot_median_rows, HISTORY_GROUP, "lot_median")
    profile = ReferenceProfile(
        version=config.reference_version,
        config_version=config.version,
        training_lot_ids=sorted(str(x) for x in frame["lot_id"].unique()),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        historical=historical,
        historical_transformed=historical_transformed,
        slopes=slope_stats,
        lot_medians=lot_medians,
        library_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    )

    features = build_feature_frame(frame, config, profile)
    profile.mahalanobis_models = _fit_mahalanobis(features)
    return profile


def _fit_mahalanobis(features: pd.DataFrame) -> dict[tuple[Any, ...], MahalanobisArtifact]:
    artifacts: dict[tuple[Any, ...], MahalanobisArtifact] = {}
    if features.empty:
        return artifacts
    context_columns = ["part_number", "time_h", "test_condition_id"]
    for context, group in features.groupby(context_columns, sort=False):
        pivot = group.pivot_table(
            index=["lot_id", "component_id"],
            columns="parameter",
            values="historical_z_signed",
            aggfunc="first",
        ).dropna(axis=1, how="all")
        pivot = pivot.dropna(axis=0, how="any")
        parameter_names = [str(x) for x in pivot.columns]
        minimum = max(30, 10 * len(parameter_names))
        if len(parameter_names) < 2 or len(pivot) < minimum:
            continue
        try:
            estimator = MinCovDet(random_state=170).fit(pivot.to_numpy(dtype=float))
            scores = estimator.mahalanobis(pivot.to_numpy(dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            continue
        artifacts[tuple(context)] = MahalanobisArtifact(
            parameters=parameter_names,
            estimator=estimator,
            training_scores=np.sort(scores).astype(float).tolist(),
        )
    return artifacts


def save_reference(profile: ReferenceProfile, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(profile, destination)


def load_reference(path: str | Path) -> ReferenceProfile:
    profile = joblib.load(Path(path))
    if not isinstance(profile, ReferenceProfile):
        raise TypeError("Artifact is not a Module A ReferenceProfile")
    return profile
