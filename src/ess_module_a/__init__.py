"""Dynamic outlier detection for ESS burn-in measurements."""

from .config import ModuleAConfig, ParameterConfig, default_config, load_config
from .engine import ModuleAEngine, fit_reference, score_lot
from .models import QAStatus

__all__ = [
    "ModuleAConfig",
    "ModuleAEngine",
    "ParameterConfig",
    "QAStatus",
    "default_config",
    "fit_reference",
    "load_config",
    "score_lot",
]

__version__ = "0.1.0"

