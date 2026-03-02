from fastapi import FastAPI, HTTPException
import logging
import numpy as np
from typing import Any

from app.services.scenario_service import ScenarioService
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ShockScenarioRequest,
    ScenarioComparisonResponse,
)
from app.config import settings

# configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("urbanx_api")

app = FastAPI(title="Z-Tracs AI Traffic Intelligence API")

# orchestration service (single instance)
service = ScenarioService()


@app.get("/", response_model=HealthResponse)
def home() -> HealthResponse:
    model_loaded = getattr(service.engine, "model", None) is not None
    return HealthResponse(status="ok", model_loaded=bool(model_loaded), sequence_length=settings.sequence_length)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    # validate request
    seq = request.sequence
    if len(seq) != settings.sequence_length:
        raise HTTPException(status_code=400, detail=f"sequence length must be {settings.sequence_length}")

    try:
        report = service.run(sequence=np.array(seq))
        baseline = report.get("baseline", {})
        preds = baseline.get("predictions", [])
        from app.risk_engine import classify_risk_value
        risk_levels = [classify_risk_value(p) for p in preds]
        overall_decision = baseline.get("decision", "")
        logger.info("/predict served", extra={"horizon": settings.horizon, "model_version": settings.model_version})
        return PredictionResponse(predictions=preds, risk_levels=risk_levels, overall_decision=overall_decision)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph-info")
def graph_info() -> dict:
    """Return information about graph propagation configuration."""
    return {
        "graph_propagation_enabled": settings.use_graph_propagation,
        "graph_alpha": settings.graph_alpha,
        "graph_iterations": settings.graph_iterations,
        "adjacency_matrix_path": settings.adjacency_matrix_path,
    }


@app.post("/scenario", response_model=ScenarioComparisonResponse)
def simulate_scenario(request: ShockScenarioRequest) -> ScenarioComparisonResponse:
    # validate
    seq = request.sequence
    if len(seq) != settings.sequence_length:
        raise HTTPException(status_code=400, detail=f"sequence length must be {settings.sequence_length}")

    try:
        from app.shock_engine import ShockParams

        shock_params = ShockParams(
            vehicle_count_increase=request.shock_params.vehicle_count_increase,
            peak_hour_multiplier=request.shock_params.peak_hour_multiplier,
            affected_time_steps=request.shock_params.affected_time_steps,
        )

        report = service.run(sequence=np.array(seq), shock_params=shock_params)

        response = ScenarioComparisonResponse(
            baseline=report.get("baseline", {}),
            shocked=report.get("shocked", {}),
            impact=report.get("impact", {}),
            risk_analysis=report.get("risk_analysis", {}),
            recommendations=report.get("recommendations", []),
            metadata=report.get("metadata", {}),
        )
        logger.info("/scenario served", extra={"shock_applied": report.get("metadata", {}).get("shock_applied", False)})
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("scenario simulation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health() -> Any:
    model_loaded = getattr(service.engine, "model", None) is not None
    return {
        "model_loaded": bool(model_loaded),
        "model_version": settings.model_version,
        "sequence_length": settings.sequence_length,
        "horizon": settings.horizon,
    }