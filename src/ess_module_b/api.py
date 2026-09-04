"""FastAPI application for Module B training-artifact inference."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

from ess_module_a.validation import DataValidationError

from .config import load_config
from .engine import ModuleBEngine


class ForecastLotRequest(BaseModel):
    measurements: list[dict[str, Any]]


def create_app(engine: ModuleBEngine | None = None) -> FastAPI:
    config_path = os.environ.get("MODULE_B_CONFIG_PATH")
    artifact_path = os.environ.get("MODULE_B_ARTIFACT_PATH")
    active_engine = engine or ModuleBEngine(load_config(config_path))
    if engine is None and artifact_path:
        active_engine = ModuleBEngine.load(artifact_path, load_config(config_path))

    app = FastAPI(
        title="ESS Module B",
        version="0.1.0",
        description="Explainable 168-hour drift forecasting from 0-hour and 24-hour values",
    )
    app.state.engine = active_engine

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_loaded": app.state.engine.artifact is not None,
        }

    @app.get("/v1/module-b/model-info")
    def model_info() -> dict[str, Any]:
        return app.state.engine.model_info()

    @app.post("/v1/module-b/forecast-lot")
    def forecast(request: ForecastLotRequest) -> dict[str, Any]:
        try:
            return app.state.engine.forecast_lot(pd.DataFrame(request.measurements))
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
