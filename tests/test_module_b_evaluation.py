from __future__ import annotations

from ess_module_a.synthetic import generate_synthetic_data
from ess_module_b.evaluation import evaluate_lot_safe


def test_module_b_lot_safe_evaluation_reports_mae_baselines_and_false_negatives():
    frame = generate_synthetic_data(n_lots=10, components_per_lot=20, seed=515)
    report = evaluate_lot_safe(frame, seed=9)
    test = report["test"]
    assert test["regression"]["count"] > 0
    assert test["regression"]["mae"] >= 0
    assert test["regression"]["persistence_mae"] >= 0
    assert test["regression"]["linear_extrapolation_mae"] >= 0
    assert "false_negatives" in test["screening"]
    assert "weighted_error_cost" in test["screening"]
