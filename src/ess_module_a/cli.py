"""Command-line entry points for generation, fitting, scoring, and serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .engine import ModuleAEngine
from .evaluation import evaluate_lot_safe
from .synthetic import generate_synthetic_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ess-module-a")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="Generate deterministic synthetic data")
    generate.add_argument("--output", default="data/synthetic.csv")
    generate.add_argument("--lots", type=int, default=30)
    generate.add_argument("--components", type=int, default=100)
    generate.add_argument("--seed", type=int, default=170)
    generate.add_argument("--quality-issues", action="store_true")

    fit = subcommands.add_parser("fit", help="Fit and save a historical reference")
    fit.add_argument("--input", required=True)
    fit.add_argument("--output", default="artifacts/reference.joblib")
    fit.add_argument("--config", default="configs/parameters.yaml")

    score = subcommands.add_parser("score", help="Score one lot")
    score.add_argument("--input", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", default="artifacts/score.json")
    score.add_argument("--config", default="configs/parameters.yaml")
    score.add_argument("--as-of", type=float, default=None)

    evaluate = subcommands.add_parser("evaluate", help="Run lot-safe labelled evaluation")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", default="artifacts/evaluation.json")
    evaluate.add_argument("--config", default="configs/parameters.yaml")
    evaluate.add_argument("--as-of", type=float, default=168.0)
    evaluate.add_argument("--seed", type=int, default=170)

    serve = subcommands.add_parser("serve", help="Run the local FastAPI service")
    serve.add_argument("--reference", required=True)
    serve.add_argument("--config", default="configs/parameters.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate":
        frame = generate_synthetic_data(
            n_lots=args.lots,
            components_per_lot=args.components,
            seed=args.seed,
            include_quality_issues=args.quality_issues,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Wrote {len(frame):,} measurements to {destination}")
        return

    config = load_config(args.config)
    if args.command == "fit":
        engine = ModuleAEngine(config)
        engine.fit(pd.read_csv(args.input))
        engine.save(args.output)
        print(json.dumps(engine.model_info(), indent=2))
        return

    if args.command == "evaluate":
        result = evaluate_lot_safe(
            pd.read_csv(args.input), config, as_of_h=args.as_of, seed=args.seed
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote evaluation report to {destination}")
        return

    engine = ModuleAEngine.load(args.reference, config)
    if args.command == "score":
        result = engine.score_lot(pd.read_csv(args.input), as_of_h=args.as_of)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote score report to {destination}")
        return

    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
