"""Command-line fitting, forecasting, evaluation, and serving for Module B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .engine import ModuleBEngine
from .evaluation import evaluate_lot_safe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ess-module-b")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="Fit and save the 168-hour predictor")
    fit.add_argument("--input", required=True)
    fit.add_argument("--output", default="artifacts/module_b_predictor.joblib")
    fit.add_argument("--config", default="configs/module_b.yaml")

    forecast = commands.add_parser("forecast", help="Forecast one lot from 0 h and 24 h")
    forecast.add_argument("--input", required=True)
    forecast.add_argument("--artifact", required=True)
    forecast.add_argument("--output", default="artifacts/module_b_forecast.json")
    forecast.add_argument("--config", default="configs/module_b.yaml")

    evaluate = commands.add_parser("evaluate", help="Run lot-safe Module B evaluation")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", default="artifacts/module_b_evaluation.json")
    evaluate.add_argument("--config", default="configs/module_b.yaml")
    evaluate.add_argument("--seed", type=int, default=170)

    serve = commands.add_parser("serve", help="Run the local Module B API")
    serve.add_argument("--artifact", required=True)
    serve.add_argument("--config", default="configs/module_b.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    return parser


def _write_json(payload, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "fit":
        engine = ModuleBEngine(config)
        engine.fit(pd.read_csv(args.input))
        engine.save(args.output)
        print(json.dumps(engine.model_info(), indent=2))
        return
    if args.command == "evaluate":
        _write_json(
            evaluate_lot_safe(pd.read_csv(args.input), config, seed=args.seed),
            args.output,
        )
        return

    engine = ModuleBEngine.load(args.artifact, config)
    if args.command == "forecast":
        _write_json(engine.forecast_lot(pd.read_csv(args.input)), args.output)
        return
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
