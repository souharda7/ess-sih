"""FastAPI application for offline Module A scoring."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, ConfigDict

from .config import load_config
from .engine import ModuleAEngine
from .validation import DataValidationError


class MeasurementIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    component_id: str
    lot_id: str
    part_number: str
    parameter: str
    time_h: float
    value: float
    unit: str
    test_condition_id: str
    temperature_c: float | None = None
    voltage_v: float | None = None
    test_mode: str | None = None
    tester_id: str | None = None
    chamber_id: str | None = None
    socket_id: str | None = None


class ScoreLotRequest(BaseModel):
    measurements: list[MeasurementIn]
    as_of_h: float | None = None


def create_app(engine: ModuleAEngine | None = None) -> FastAPI:
    config_path = os.environ.get("MODULE_A_CONFIG_PATH")
    reference_path = os.environ.get("MODULE_A_REFERENCE_PATH")
    active_engine = engine or ModuleAEngine(load_config(config_path))
    if engine is None and reference_path:
        active_engine = ModuleAEngine.load(reference_path, load_config(config_path))

    app = FastAPI(
        title="ESS Module A",
        version="0.1.0",
        description="Explainable dynamic outlier detection for burn-in lots",
    )
    app.state.engine = active_engine

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "reference_loaded": app.state.engine.reference is not None,
        }

    @app.get("/v1/module-a/model-info")
    def model_info() -> dict[str, Any]:
        return app.state.engine.model_info()

    @app.post("/v1/module-a/score-lot")
    def score_lot(request: ScoreLotRequest) -> dict[str, Any]:
        frame = pd.DataFrame([measurement.model_dump() for measurement in request.measurements])
        try:
            return app.state.engine.score_lot(frame, as_of_h=request.as_of_h)
        except DataValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
