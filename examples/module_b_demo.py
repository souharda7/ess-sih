"""Fit Module B, forecast an unseen lot at 24 hours, and print the evidence."""

from __future__ import annotations

from collections import Counter

from ess_module_a.synthetic import generate_synthetic_data
from ess_module_b import ModuleBEngine, default_config


def main() -> None:
    training = generate_synthetic_data(n_lots=12, components_per_lot=60, seed=170)
    engine = ModuleBEngine(default_config())
    engine.fit(training)

    unseen = generate_synthetic_data(n_lots=3, components_per_lot=60, seed=171)
    lot = unseen.loc[unseen["lot_id"] == "LOT_001"].copy()
    lot["lot_id"] = "MODULE_B_DEMO"
    lot["component_id"] = lot["component_id"].str.replace("LOT_001", "MODULE_B_DEMO")
    report = engine.forecast_lot(lot)

    counts = Counter(item["decision"] for item in report["component_results"])
    print("Component decisions:", dict(counts))
    print("\nEarly rejections:")
    for item in report["prediction_results"]:
        if item["flagged_for_early_rejection"]:
            print(item["component_id"], item["parameter"], item["explanation"]["summary"])


if __name__ == "__main__":
    main()
