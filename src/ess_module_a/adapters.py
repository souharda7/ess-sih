"""Adapters for external and organizer-provided measurement layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class NASALeakageSnapshot:
    """Metadata needed to turn one NASA LeakageIV curve into one ESS reading."""

    component_id: str
    time_h: float
    path: str | Path


def adapt_nasa_igbt_leakage(
    snapshots: list[NASALeakageSnapshot],
    *,
    target_voltage_v: float,
    lot_id: str = "NASA_IGBT_SANITY",
    part_number: str = "IGBT_IRG4BC30K",
    temperature_c: float = 25.0,
) -> pd.DataFrame:
    """Convert NASA two-column LeakageIV curves to Module A long-form rows.

    The public archive stores applied voltage in column 1 and measured leakage
    current in amperes in column 2. The nearest point to ``target_voltage_v``
    is selected. This adapter is intentionally a pipeline sanity check: the
    NASA aging collection is too small to establish production-lot thresholds.
    """

    records = []
    for snapshot in snapshots:
        curve = pd.read_csv(
            snapshot.path,
            header=None,
            names=["applied_voltage_v", "leakage_current_a"],
        )
        curve["applied_voltage_v"] = pd.to_numeric(curve["applied_voltage_v"], errors="coerce")
        curve["leakage_current_a"] = pd.to_numeric(curve["leakage_current_a"], errors="coerce")
        curve = curve.dropna()
        if curve.empty:
            raise ValueError(f"No numeric LeakageIV samples in {snapshot.path}")
        nearest_index = (curve["applied_voltage_v"] - target_voltage_v).abs().idxmin()
        sample = curve.loc[nearest_index]
        records.append(
            {
                "component_id": snapshot.component_id,
                "lot_id": lot_id,
                "part_number": part_number,
                "parameter": "leakage_current",
                "time_h": float(snapshot.time_h),
                "value": float(sample["leakage_current_a"]),
                "unit": "A",
                "test_condition_id": f"NASA_IGBT_{target_voltage_v:g}V",
                "temperature_c": float(temperature_c),
                "voltage_v": float(sample["applied_voltage_v"]),
                "test_mode": "leakage_iv_nearest_point",
                "tester_id": "NASA_PCOE",
                "chamber_id": None,
                "socket_id": None,
                "source_file": str(snapshot.path),
                "requested_voltage_v": float(target_voltage_v),
            }
        )
    return pd.DataFrame.from_records(records)


def adapt_wide_checkpoints(
    frame: pd.DataFrame,
    *,
    id_columns: list[str],
    checkpoint_columns: dict[str, float],
    parameter: str,
    unit: str,
) -> pd.DataFrame:
    """Turn Value_0h/Value_24h-style organizer data into long-form rows."""

    missing = (set(id_columns) | set(checkpoint_columns)) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing adapter columns: {sorted(missing)}")
    melted = frame.melt(
        id_vars=id_columns,
        value_vars=list(checkpoint_columns),
        var_name="checkpoint_source",
        value_name="value",
    )
    melted["time_h"] = melted["checkpoint_source"].map(checkpoint_columns).astype(float)
    melted["parameter"] = parameter
    melted["unit"] = unit
    return melted.drop(columns=["checkpoint_source"])
