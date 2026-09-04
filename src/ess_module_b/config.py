"""Versioned model-selection and engineering-safety configuration for Module B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml

from ess_module_a.config import ModuleAConfig, ParameterConfig, default_config as module_a_defaults


@dataclass(slots=True)
class ModuleBConfig:
    version: str = "1.0.0"
    artifact_version: str = "1.0.0"
    baseline_h: float = 0.0
    early_h: float = 24.0
    target_h: float = 168.0
    minimum_training_samples: int = 30
    minimum_training_lots: int = 3
    cv_folds: int = 5
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    huber_epsilon: float = 1.35
    safety_quantile: float = 0.99
    uncertainty_quantile: float = 0.95
    linear_sentinel_multiplier: float = 4.5
    guard_band_fraction: float = 0.05
    false_negative_cost: float = 10.0
    epsilon: float = 1e-9
    random_seed: int = 170
    parameters: dict[str, ParameterConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.baseline_h < self.early_h < self.target_h:
            raise ValueError("Module B checkpoints must satisfy baseline_h < early_h < target_h")
        if self.minimum_training_samples < 2:
            raise ValueError("minimum_training_samples must be at least 2")
        if self.minimum_training_lots < 2:
            raise ValueError("minimum_training_lots must be at least 2")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if not self.ridge_alphas or any(alpha <= 0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain positive values")
        for name, value in {
            "safety_quantile": self.safety_quantile,
            "uncertainty_quantile": self.uncertainty_quantile,
        }.items():
            if not 0.5 < value < 1.0:
                raise ValueError(f"{name} must be between 0.5 and 1.0")
        if not 0.0 <= self.guard_band_fraction < 0.5:
            raise ValueError("guard_band_fraction must be in [0, 0.5)")
        if self.linear_sentinel_multiplier < 1.0:
            raise ValueError("linear_sentinel_multiplier must be at least 1.0")
        if not self.parameters:
            self.parameters = module_a_defaults().parameters

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModuleBConfig":
        item = dict(value)
        item["ridge_alphas"] = tuple(
            float(x) for x in item.get("ridge_alphas", (0.01, 0.1, 1.0, 10.0))
        )
        raw_parameters = item.get("parameters")
        if raw_parameters is not None:
            item["parameters"] = {
                key: ParameterConfig.from_dict({"parameter": key, **parameter})
                for key, parameter in raw_parameters.items()
            }
        return cls(**item)

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["ridge_alphas"] = list(self.ridge_alphas)
        item["parameters"] = {}
        for key, parameter in self.parameters.items():
            payload = parameter.to_dict()
            payload.pop("parameter", None)
            item["parameters"][key] = payload
        return item

    def validation_config(self, checkpoints: tuple[float, ...]) -> ModuleAConfig:
        """Use Module A's proven validation/unit-normalization layer with B checkpoints."""

        return ModuleAConfig(
            version=self.version,
            reference_version=self.artifact_version,
            checkpoints_h=checkpoints,
            epsilon=self.epsilon,
            random_seed=self.random_seed,
            parameters=self.parameters,
        )


def default_config() -> ModuleBConfig:
    return ModuleBConfig(parameters=module_a_defaults().parameters)


def load_config(path: str | Path | None = None) -> ModuleBConfig:
    if path is None:
        return default_config()
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) if source.suffix.lower() == ".json" else yaml.safe_load(handle)
    return ModuleBConfig.from_dict(payload)
