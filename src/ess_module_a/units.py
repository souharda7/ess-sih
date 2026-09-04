"""Small, explicit unit converter for supported ESS parameters."""

from __future__ import annotations

import re


_UNIT_ALIASES = {
    "µa": "uA",
    "μa": "uA",
    "ua": "uA",
    "ma": "mA",
    "na": "nA",
    "pa": "pA",
    "a": "A",
    "v": "V",
    "mv": "mV",
    "uv": "uV",
    "µv": "uV",
    "μv": "uV",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "µs": "us",
    "μs": "us",
    "ns": "ns",
    "ps": "ps",
}

_UNIT_TABLE = {
    "A": ("current", 1.0),
    "mA": ("current", 1e-3),
    "uA": ("current", 1e-6),
    "nA": ("current", 1e-9),
    "pA": ("current", 1e-12),
    "V": ("voltage", 1.0),
    "mV": ("voltage", 1e-3),
    "uV": ("voltage", 1e-6),
    "s": ("time", 1.0),
    "ms": ("time", 1e-3),
    "us": ("time", 1e-6),
    "ns": ("time", 1e-9),
    "ps": ("time", 1e-12),
}


def normalize_unit(unit: str) -> str:
    compact = re.sub(r"\s+", "", str(unit)).lower()
    try:
        return _UNIT_ALIASES[compact]
    except KeyError as exc:
        raise ValueError(f"Unsupported unit: {unit}") from exc


def convert_value(value: float, source_unit: str, target_unit: str) -> float:
    source = normalize_unit(source_unit)
    target = normalize_unit(target_unit)
    source_dimension, source_scale = _UNIT_TABLE[source]
    target_dimension, target_scale = _UNIT_TABLE[target]
    if source_dimension != target_dimension:
        raise ValueError(f"Cannot convert {source_unit} to {target_unit}")
    return float(value) * source_scale / target_scale

