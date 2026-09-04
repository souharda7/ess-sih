"""Lot-safe evaluation utilities for labelled development data."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import ModuleAConfig, default_config
from .engine import ModuleAEngine


FLAGGED_STATUSES = {"MONITOR", "QUARANTINE", "STATIC_FAIL", "RETEST_REQUIRED"}


def split_lots(
    measurements: pd.DataFrame,
    *,
    seed: int = 170,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, pd.DataFrame]:
    lots = np.asarray(sorted(str(x) for x in measurements["lot_id"].unique()))
    if len(lots) < 5:
        raise ValueError("At least five lots are required for train/validation/test splitting")
    rng = np.random.default_rng(seed)
    rng.shuffle(lots)
    train_end = max(1, int(len(lots) * train_fraction))
    validation_end = max(train_end + 1, int(len(lots) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(lots) - 1)
    partitions = {
        "train": set(lots[:train_end]),
        "validation": set(lots[train_end:validation_end]),
        "test": set(lots[validation_end:]),
    }
    return {
        name: measurements.loc[measurements["lot_id"].astype(str).isin(lot_ids)].copy()
        for name, lot_ids in partitions.items()
    }


def evaluate_partition(
    engine: ModuleAEngine,
    measurements: pd.DataFrame,
    *,
    as_of_h: float = 168.0,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    lot_recalls: dict[str, float] = {}
    for lot_id, lot in measurements.groupby("lot_id", sort=False):
        report = engine.score_lot(lot, as_of_h=as_of_h)
        labels = _component_labels(lot)
        component_predictions = []
        for item in report["component_results"]:
            label = labels[item["component_id"]]
            record = {
                "lot_id": str(lot_id),
                "component_id": item["component_id"],
                "actual": bool(label["is_anomaly"]),
                "defect_type": label["defect_type"],
                "predicted": item["status"] in FLAGGED_STATUSES,
                "status": item["status"],
                "risk_score": float(item["risk_score"]),
            }
            component_predictions.append(record)
            predictions.append(record)
        positives = [item for item in component_predictions if item["actual"]]
        if positives:
            lot_recalls[str(lot_id)] = sum(item["predicted"] for item in positives) / len(positives)

    actual = np.asarray([item["actual"] for item in predictions], dtype=bool)
    predicted = np.asarray([item["predicted"] for item in predictions], dtype=bool)
    scores = np.asarray([item["risk_score"] for item in predictions], dtype=float)
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    defect_recall = tp / (tp + fn) if tp + fn else None
    false_flag_rate = fp / (fp + tn) if fp + tn else None
    pr_auc = float(average_precision_score(actual, scores)) if len(np.unique(actual)) > 1 else None

    by_defect: dict[str, dict[str, int | float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        if item["actual"]:
            grouped[item["defect_type"]].append(item)
    for defect_type, items in grouped.items():
        detected = sum(item["predicted"] for item in items)
        by_defect[defect_type] = {
            "count": len(items),
            "detected": detected,
            "recall": detected / len(items),
        }

    return {
        "as_of_h": as_of_h,
        "component_count": len(predictions),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "defect_recall": defect_recall,
        "false_flag_rate": false_flag_rate,
        "precision_recall_auc": pr_auc,
        "worst_lot_recall": min(lot_recalls.values()) if lot_recalls else None,
        "recall_by_defect_type": by_defect,
        "status_counts": pd.Series([item["status"] for item in predictions]).value_counts().to_dict(),
    }


def evaluate_lot_safe(
    measurements: pd.DataFrame,
    config: ModuleAConfig | None = None,
    *,
    as_of_h: float = 168.0,
    seed: int = 170,
) -> dict[str, Any]:
    active_config = config or default_config()
    partitions = split_lots(measurements, seed=seed)
    engine = ModuleAEngine(active_config)
    engine.fit(partitions["train"])
    return {
        "split_lots": {
            name: sorted(str(x) for x in frame["lot_id"].unique())
            for name, frame in partitions.items()
        },
        "model_info": engine.model_info(),
        "validation": evaluate_partition(engine, partitions["validation"], as_of_h=as_of_h),
        "test": evaluate_partition(engine, partitions["test"], as_of_h=as_of_h),
    }


def _component_labels(lot: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for component_id, group in lot.groupby("component_id", sort=False):
        anomaly = bool(group.get("is_anomaly", pd.Series(False, index=group.index)).fillna(False).any())
        defect_values = [
            str(value)
            for value in group.get("defect_type", pd.Series("normal", index=group.index)).dropna().unique()
            if str(value) != "normal"
        ]
        output[str(component_id)] = {
            "is_anomaly": anomaly,
            "defect_type": "+".join(sorted(defect_values)) if defect_values else "normal",
        }
    return output
