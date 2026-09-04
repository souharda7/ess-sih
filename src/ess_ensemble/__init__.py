"""Safety-first fusion of Module A observations and Module B forecasts."""

from .engine import EnsembleDecision, EnsembleEngine, combine_reports
from .evaluation import evaluate_lot_safe, evaluate_partition

__all__ = [
    "EnsembleDecision",
    "EnsembleEngine",
    "combine_reports",
    "evaluate_lot_safe",
    "evaluate_partition",
]

__version__ = "0.1.0"
