from __future__ import annotations

from ess_module_a.evaluation import evaluate_lot_safe, split_lots
from ess_module_a.synthetic import generate_synthetic_data


def test_lot_split_has_no_overlap():
    frame = generate_synthetic_data(n_lots=10, components_per_lot=20, seed=400)
    parts = split_lots(frame, seed=3)
    lot_sets = {name: set(part["lot_id"].unique()) for name, part in parts.items()}
    assert lot_sets["train"].isdisjoint(lot_sets["validation"])
    assert lot_sets["train"].isdisjoint(lot_sets["test"])
    assert lot_sets["validation"].isdisjoint(lot_sets["test"])


def test_lot_safe_evaluation_reports_safety_metrics():
    frame = generate_synthetic_data(n_lots=10, components_per_lot=20, seed=401)
    report = evaluate_lot_safe(frame, as_of_h=24, seed=5)
    assert report["test"]["component_count"] > 0
    assert "false_negatives" in report["test"]
    assert "defect_recall" in report["test"]
    assert "worst_lot_recall" in report["test"]
