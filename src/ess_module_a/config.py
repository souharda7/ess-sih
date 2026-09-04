"""Versioned engineering and detector configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml

from .models import DangerDirection


@dataclass(slots=True)
class ParameterConfig:
    parameter: str
    canonical_unit: str
    danger_direction: DangerDirection
    spec_min: float | None = None
    spec_max: float | None = None
    transform: str = "raw"
    minimum_lot_peers: int = 10
    minimum_historical_peers: int = 30
    required_context_fields: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParameterConfig":
        item = dict(value)
        item["danger_direction"] = DangerDirection(item["danger_direction"])
        item["required_context_fields"] = tuple(item.get("required_context_fields", ()))
        return cls(**item)

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["danger_direction"] = self.danger_direction.value
        item["required_context_fields"] = list(self.required_context_fields)
        return item


@dataclass(slots=True)
class ModuleAConfig:
    version: str = "1.0.0"
    reference_version: str = "1.0.0"
    checkpoints_h: tuple[float, ...] = (0.0, 24.0, 96.0, 168.0)
    robust_z_warning: float = 2.5
    robust_z_severe: float = 3.5
    robust_z_extreme: float = 5.0
    iqr_warning_multiplier: float = 1.5
    iqr_severe_multiplier: float = 3.0
    tail_warning_percentile: float = 97.5
    tail_severe_percentile: float = 99.5
    epsilon: float = 1e-9
    random_seed: int = 170
    parameters: dict[str, ParameterConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModuleAConfig":
        item = dict(value)
        item["checkpoints_h"] = tuple(float(x) for x in item.get("checkpoints_h", (0, 24, 96, 168)))
        item["parameters"] = {
            key: ParameterConfig.from_dict({"parameter": key, **parameter})
            for key, parameter in item.get("parameters", {}).items()
        }
        return cls(**item)

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["checkpoints_h"] = list(self.checkpoints_h)
        item["parameters"] = {
            key: parameter.to_dict() | {"parameter": key}
            for key, parameter in self.parameters.items()
        }
        for parameter in item["parameters"].values():
            parameter.pop("parameter", None)
        return item


def default_config() -> ModuleAConfig:
    return ModuleAConfig(
        parameters={
            "leakage_current": ParameterConfig(
                parameter="leakage_current",
                canonical_unit="uA",
                danger_direction=DangerDirection.HIGHER,
                spec_max=50.0,
                transform="log1p",
                required_context_fields=("temperature_c", "voltage_v", "test_mode"),
            ),
            "iddq": ParameterConfig(
                parameter="iddq",
                canonical_unit="uA",
                danger_direction=DangerDirection.HIGHER,
                spec_max=80.0,
                transform="log1p",
                required_context_fields=("temperature_c", "voltage_v", "test_mode"),
            ),
            "propagation_delay": ParameterConfig(
                parameter="propagation_delay",
                canonical_unit="ns",
                danger_direction=DangerDirection.HIGHER,
                spec_max=20.0,
                required_context_fields=("temperature_c", "voltage_v", "test_mode"),
            ),
            "output_high_voltage": ParameterConfig(
                parameter="output_high_voltage",
                canonical_unit="V",
                danger_direction=DangerDirection.LOWER,
                spec_min=2.4,
                required_context_fields=("temperature_c", "voltage_v", "test_mode"),
            ),
            "threshold_voltage": ParameterConfig(
                parameter="threshold_voltage",
                canonical_unit="V",
                danger_direction=DangerDirection.TWO_SIDED,
                spec_min=0.35,
                spec_max=0.95,
                required_context_fields=("temperature_c", "voltage_v", "test_mode"),
            ),
        }
    )


def load_config(path: str | Path | None = None) -> ModuleAConfig:
    if path is None:
        return default_config()
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    return ModuleAConfig.from_dict(payload)

