"""Generate training lots, fit Module A, and score one unseen lot at 24 hours."""

from __future__ import annotations

from collections import Counter

from ess_module_a import ModuleAEngine, default_config
from ess_module_a.synthetic import generate_synthetic_data


def main() -> None:
    training = generate_synthetic_data(n_lots=12, components_per_lot=60, seed=170)
    engine = ModuleAEngine(default_config())
    engine.fit(training)

    unseen = generate_synthetic_data(n_lots=3, components_per_lot=60, seed=171)
    lot = unseen.loc[(unseen["lot_id"] == "LOT_001") & (unseen["time_h"] <= 24)].copy()
    lot["lot_id"] = "DEMO_LOT"
    lot["component_id"] = lot["component_id"].str.replace("LOT_001", "DEMO_LOT")
    report = engine.score_lot(lot, as_of_h=24)

    counts = Counter(item["status"] for item in report["component_results"])
    print("Component statuses:", dict(counts))
    print("Lot alerts:", report["lot_alerts"])
    print("\nFlagged components:")
    for item in report["component_results"]:
        if item["status"] != "NORMAL":
            print(item)


if __name__ == "__main__":
    main()
