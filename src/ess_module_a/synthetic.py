"""Transparent synthetic ESS lot generator for development and evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


PARAMETERS = {
    "leakage_current": "uA",
    "iddq": "uA",
    "propagation_delay": "ns",
    "output_high_voltage": "V",
    "threshold_voltage": "V",
}

CHECKPOINTS = (0.0, 24.0, 96.0, 168.0)


def generate_synthetic_data(
    *,
    n_lots: int = 30,
    components_per_lot: int = 100,
    seed: int = 170,
    defect_fraction: float = 0.05,
    include_quality_issues: bool = False,
) -> pd.DataFrame:
    if n_lots < 3:
        raise ValueError("At least three lots are required")
    if components_per_lot < 20:
        raise ValueError("At least 20 components per lot are required")
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    defect_cycle = ["high_in_spec", "fast_drift", "accelerating", "correlated", "static_fail"]
    whole_lot_index = n_lots - 1
    equipment_lot_index = n_lots - 2

    family_bases = {
        "PN_LOGIC_A": {
            "leakage_current": 10.0,
            "iddq": 22.0,
            "propagation_delay": 5.0,
            "output_high_voltage": 3.20,
            "threshold_voltage": 0.62,
            "voltage_v": 3.3,
        },
        "PN_LOGIC_B": {
            "leakage_current": 14.0,
            "iddq": 30.0,
            "propagation_delay": 7.0,
            "output_high_voltage": 4.85,
            "threshold_voltage": 0.70,
            "voltage_v": 5.0,
        },
    }

    # Hard cap: at most defect_fraction of components can be defective per lot.
    max_defects = max(1, int(components_per_lot * defect_fraction))

    for lot_index in range(n_lots):
        lot_id = f"LOT_{lot_index + 1:03d}"
        part_number = "PN_LOGIC_A" if lot_index % 2 == 0 else "PN_LOGIC_B"
        base = family_bases[part_number]
        # Wider lot-to-lot process shift to produce realistic inter-lot fluctuations.
        lot_factor = float(np.exp(rng.normal(0.0, 0.06)))
        lot_additive = float(rng.normal(0.0, 0.022))
        # Random defect count per lot: anywhere from 0 up to max_defects (inclusive).
        defect_count = int(rng.integers(0, max_defects + 1))
        if defect_count == 0:
            defective_indexes: set[int] = set()
            defect_names: dict[int, str] = {}
        else:
            defective_indexes = set(
                int(x) for x in rng.choice(components_per_lot, size=defect_count, replace=False)
            )
            defect_names = {
                index: defect_cycle[(lot_index * defect_count + position) % len(defect_cycle)]
                for position, index in enumerate(sorted(defective_indexes))
            }


        for component_index in range(components_per_lot):
            component_id = f"{lot_id}_C{component_index + 1:04d}"
            # Wider per-component noise so healthy parts show realistic spread.
            component_factor = float(np.exp(rng.normal(0.0, 0.10)))
            component_additive = float(rng.normal(0.0, 0.018))
            tester_id = f"T{1 + component_index % 3}"
            chamber_id = f"CH{1 + lot_index % 2}"
            defect_type = defect_names.get(component_index, "normal")
            if lot_index == whole_lot_index:
                defect_type = "whole_lot_shift"
            elif lot_index == equipment_lot_index and tester_id == "T3":
                defect_type = "tester_offset"

            for time_h in CHECKPOINTS:
                fraction = time_h / 168.0
                values = {
                    "leakage_current": base["leakage_current"]
                    * lot_factor
                    * component_factor
                    * (1.0 + 0.04 * fraction)
                    * float(np.exp(rng.normal(0.0, 0.018))),
                    "iddq": base["iddq"]
                    * lot_factor
                    * component_factor
                    * (1.0 + 0.03 * fraction)
                    * float(np.exp(rng.normal(0.0, 0.015))),
                    "propagation_delay": base["propagation_delay"]
                    * lot_factor
                    * component_factor
                    + 0.20 * fraction
                    + float(rng.normal(0.0, 0.05)),
                    "output_high_voltage": base["output_high_voltage"]
                    + lot_additive
                    + component_additive
                    - 0.03 * fraction
                    + float(rng.normal(0.0, 0.008)),
                    "threshold_voltage": base["threshold_voltage"]
                    + lot_additive
                    + component_additive
                    + 0.01 * fraction
                    + float(rng.normal(0.0, 0.006)),
                }

                _inject_component_defect(values, defect_type, time_h)
                if defect_type == "whole_lot_shift":
                    values["leakage_current"] *= 2.0
                    values["iddq"] *= 1.7
                    values["propagation_delay"] *= 1.18
                    values["output_high_voltage"] -= 0.18
                elif defect_type == "tester_offset":
                    values["leakage_current"] *= 1.45
                    values["iddq"] *= 1.30
                    values["propagation_delay"] *= 1.08

                for parameter, value in values.items():
                    records.append(
                        {
                            "component_id": component_id,
                            "lot_id": lot_id,
                            "part_number": part_number,
                            "parameter": parameter,
                            "time_h": time_h,
                            "value": round(float(value), 8),
                            "unit": PARAMETERS[parameter],
                            "test_condition_id": f"{part_number}_125C_NOMINAL",
                            "temperature_c": 125.0,
                            "voltage_v": base["voltage_v"],
                            "test_mode": "static_bias",
                            "tester_id": tester_id,
                            "chamber_id": chamber_id,
                            "socket_id": f"S{1 + component_index % 20:02d}",
                            "is_anomaly": defect_type != "normal",
                            "defect_type": defect_type,
                            "qa_approved": defect_type == "normal",
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    if include_quality_issues:
        frame = _inject_quality_issues(frame)
    return frame.reset_index(drop=True)


def _inject_component_defect(values: dict[str, float], defect_type: str, time_h: float) -> None:
    fraction = time_h / 168.0
    if defect_type == "high_in_spec" and time_h >= 24:
        values["leakage_current"] = 45.0
    elif defect_type == "fast_drift" and time_h >= 24:
        values["leakage_current"] += 44.0 * fraction
    elif defect_type == "accelerating":
        values["leakage_current"] += 44.0 * fraction**2
        values["propagation_delay"] += 4.0 * fraction**2
    elif defect_type == "correlated":
        values["leakage_current"] += 20.0 * fraction
        values["iddq"] += 35.0 * fraction
        values["propagation_delay"] += 5.0 * fraction
        values["output_high_voltage"] -= 0.35 * fraction
    elif defect_type == "static_fail" and time_h >= 24:
        values["leakage_current"] = 60.0 + 5.0 * fraction


def _inject_quality_issues(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    components = result["component_id"].drop_duplicates().tolist()
    if len(components) < 3:
        return result

    missing_component = components[0]
    missing_mask = (
        (result["component_id"] == missing_component)
        & (result["parameter"] == "leakage_current")
        & (result["time_h"] == 24)
    )
    result = result.loc[~missing_mask].copy()

    duplicate_component = components[1]
    duplicate_row = result.loc[
        (result["component_id"] == duplicate_component)
        & (result["parameter"] == "iddq")
        & (result["time_h"] == 24)
    ].head(1)
    result = pd.concat([result, duplicate_row], ignore_index=True)

    bad_unit_component = components[2]
    bad_unit_mask = (
        (result["component_id"] == bad_unit_component)
        & (result["parameter"] == "propagation_delay")
        & (result["time_h"] == 24)
    )
    result.loc[bad_unit_mask, "unit"] = "furlong"
    result.loc[bad_unit_mask, "qa_approved"] = False
    result.loc[bad_unit_mask, "defect_type"] = "invalid_unit"
    return result
