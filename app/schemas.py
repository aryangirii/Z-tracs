from typing import List, Optional, Dict, Any
from pydantic import BaseModel, conlist


# ----------------------------
# Prediction
# ----------------------------

class PredictionRequest(BaseModel):
    # sequence of timesteps; each timestep is a list of floats
    sequence: List[conlist(float, min_length=1)]


class PredictionResponse(BaseModel):
    predictions: List[float]
    risk_levels: List[str]
    overall_decision: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    sequence_length: int


# ----------------------------
# Shock / Scenario
# ----------------------------

class ShockParamsRequest(BaseModel):
    vehicle_count_increase: float = 0.0
    peak_hour_multiplier: float = 1.0
    affected_time_steps: Optional[List[int]] = None


class ShockScenarioRequest(BaseModel):
    sequence: List[conlist(float, min_length=1)]
    shock_params: ShockParamsRequest


class ScenarioComparisonResponse(BaseModel):
    baseline: Dict[str, Any]
    shocked: Dict[str, Any]
    impact: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    recommendations: List[str]