"""Lot-safe model selection, calibration, and artifact persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModuleBConfig
from .data import prepare_series
from .features import (
    FEATURE_NAMES,
    direction_aware_slope,
    engineer_features,
    linear_extrapolation_prediction,
    persistence_prediction,
)
from .models import DriftPredictorArtifact, FittedDriftModel


CONTEXT_COLUMNS = ["part_number", "parameter", "test_condition_id"]


def _candidate_estimators(config: ModuleBConfig) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {}
    for alpha in config.ridge_alphas:
        candidates[f"ridge_alpha_{alpha:g}"] = Pipeline(
            [
                ("scale", StandardScaler()),
                ("regressor", Ridge(alpha=alpha)),
            ]
        )
    candidates["huber"] = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regressor",
                HuberRegressor(
                    epsilon=config.huber_epsilon,
                    alpha=0.0001,
                    max_iter=1000,
                ),
            ),
        ]
    )
    return candidates


def _cross_validated_predictions(
    estimator: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
) -> np.ndarray:
    predictions = np.full(len(y), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    for train_index, validation_index in splitter.split(x, y, groups):
        fitted = clone(estimator)
        fitted.fit(x[train_index], y[train_index])
        predictions[validation_index] = fitted.predict(x[validation_index])
    if not np.isfinite(predictions).all():
        raise ValueError("Cross-validation did not produce a finite prediction for every sample")
    return predictions


def _fit_group(group: pd.DataFrame, config: ModuleBConfig) -> FittedDriftModel | None:
    group = group.loc[
        group["data_quality_status"].eq("VALID")
        & group[["value_0h", "value_24h", "actual_value_168h"]].notna().all(axis=1)
    ].copy()
    if len(group) < config.minimum_training_samples:
        return None
    lot_count = int(group["lot_id"].nunique())
    if lot_count < config.minimum_training_lots:
        return None

    features = engineer_features(
        group,
        baseline_h=config.baseline_h,
        early_h=config.early_h,
        epsilon=config.epsilon,
    )
    x = features[FEATURE_NAMES].to_numpy(dtype=float)
    y = group["actual_value_168h"].to_numpy(dtype=float)
    groups = group["lot_id"].astype(str).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None

    candidate_mae: dict[str, float] = {}
    candidate_predictions: dict[str, np.ndarray] = {}
    candidates = _candidate_estimators(config)
    for name, estimator in candidates.items():
        try:
            predictions = _cross_validated_predictions(
                estimator, x, y, groups, config.cv_folds
            )
        except (ValueError, FloatingPointError):
            continue
        candidate_predictions[name] = predictions
        candidate_mae[name] = float(mean_absolute_error(y, predictions))
    if not candidate_mae:
        return None

    # Stable tie-breaking favors Ridge because its closed-form behavior is especially auditable.
    selected_name = min(
        candidate_mae,
        key=lambda name: (candidate_mae[name], 1 if name == "huber" else 0, name),
    )
    selected = clone(candidates[selected_name])
    selected.fit(x, y)
    selected_oof = candidate_predictions[selected_name]

    parameter_name = str(group["parameter"].iloc[0])
    parameter = config.parameters[parameter_name]
    residual = y - selected_oof
    if parameter.danger_direction.value == "higher":
        danger_residual = residual
    elif parameter.danger_direction.value == "lower":
        danger_residual = -residual
    else:
        danger_residual = np.abs(residual)
    residual_margin = max(
        0.0, float(np.quantile(danger_residual, config.uncertainty_quantile))
    )

    good = group.loc[group["is_good_reference"].fillna(False).astype(bool)]
    if good.empty:
        good = group
    total_elapsed = config.target_h - config.baseline_h
    slopes = (
        good["actual_value_168h"].to_numpy(dtype=float)
        - good["value_0h"].to_numpy(dtype=float)
    ) / total_elapsed
    danger_slopes = np.maximum(
        np.asarray(direction_aware_slope(slopes, parameter.danger_direction), dtype=float),
        0.0,
    )
    historical_safety_slope = max(
        config.epsilon,
        float(np.quantile(danger_slopes, config.safety_quantile)),
    )

    persistence = persistence_prediction(features)
    linear = linear_extrapolation_prediction(
        features,
        baseline_h=config.baseline_h,
        early_h=config.early_h,
        target_h=config.target_h,
    )
    return FittedDriftModel(
        name=selected_name,
        estimator=selected,
        feature_names=list(FEATURE_NAMES),
        training_samples=len(group),
        training_lots=lot_count,
        cv_mae=candidate_mae[selected_name],
        persistence_mae=float(mean_absolute_error(y, persistence)),
        linear_extrapolation_mae=float(mean_absolute_error(y, linear)),
        candidate_mae=candidate_mae,
        danger_residual_margin=residual_margin,
        historical_safety_slope=historical_safety_slope,
        target_min=float(np.min(y)),
        target_max=float(np.max(y)),
    )


def fit_predictor_artifact(
    measurements: pd.DataFrame, config: ModuleBConfig
) -> DriftPredictorArtifact:
    prepared = prepare_series(
        measurements,
        config,
        include_target=True,
        require_single_lot=False,
    )
    series = prepared.series
    if series.empty:
        raise ValueError("No component series could be constructed from the training data")
    valid = series.loc[
        series["data_quality_status"].eq("VALID")
        & series[["value_0h", "value_24h", "actual_value_168h"]].notna().all(axis=1)
        & series["parameter"].isin(config.parameters)
    ].copy()
    if valid.empty:
        raise ValueError("No valid complete 0 h/24 h/168 h series are available for training")

    context_models: dict[tuple[str, str, str], FittedDriftModel] = {}
    for key, group in valid.groupby(CONTEXT_COLUMNS, sort=False):
        normalized_key = tuple(str(value) for value in key)
        fitted = _fit_group(group, config)
        if fitted is not None:
            context_models[normalized_key] = fitted

    parameter_models: dict[str, FittedDriftModel] = {}
    for parameter_name, group in valid.groupby("parameter", sort=False):
        fitted = _fit_group(group, config)
        if fitted is not None:
            parameter_models[str(parameter_name)] = fitted

    if not context_models and not parameter_models:
        raise ValueError(
            "Training data does not meet minimum sample/lot requirements for any model"
        )
    return DriftPredictorArtifact(
        version=config.artifact_version,
        config_version=config.version,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        training_lot_ids=sorted(str(value) for value in valid["lot_id"].unique()),
        context_models=context_models,
        parameter_models=parameter_models,
        library_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    )


def save_artifact(artifact: DriftPredictorArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)


def load_artifact(path: str | Path) -> DriftPredictorArtifact:
    artifact: Any = joblib.load(Path(path))
    if not isinstance(artifact, DriftPredictorArtifact):
        raise TypeError("Artifact is not a Module B DriftPredictorArtifact")
    return artifact
