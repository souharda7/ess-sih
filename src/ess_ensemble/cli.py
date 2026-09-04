"""Command-line evaluation for the final Module A + Module B ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ess_module_a.config import load_config as load_module_a_config
from ess_module_b.config import load_config as load_module_b_config

from .evaluation import evaluate_lot_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Long-form labelled ESS CSV")
    parser.add_argument(
        "--output",
        default="artifacts/ensemble_acceptance_evaluation.json",
        help="Destination JSON report",
    )
    parser.add_argument(
        "--module-a-config",
        default="configs/parameters.yaml",
    )
    parser.add_argument(
        "--module-b-config",
        default="configs/module_b.yaml",
    )
    parser.add_argument("--seed", type=int, default=170)
    parser.add_argument("--as-of", type=float, default=24.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = evaluate_lot_safe(
        pd.read_csv(args.input),
        load_module_a_config(args.module_a_config),
        load_module_b_config(args.module_b_config),
        seed=args.seed,
        as_of_h=args.as_of,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
