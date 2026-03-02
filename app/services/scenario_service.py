from typing import Any, Dict, Optional
import logging
import numpy as np

from app.forecasting import ForecastEngine
from app.shock_engine import ShockSimulator, ShockParams, ComparisonResult
from app.graph_engine import GraphEngine
from app.config import settings
from app.risk_engine import classify_risk_series
from app.decision_engine import infrastructure_decision


logger = logging.getLogger("scenario_service")


class ScenarioService:
    """Orchestration layer for running baseline and shocked scenarios.

    Responsibilities:
      - Run baseline forecast
      - Apply shock to inputs
      - Run shocked forecast
      - Optionally apply spatial propagation
      - Compute deltas, severity and risk shifts
      - Produce structured report (without HTTP concerns)
    """

    def __init__(self) -> None:
        self.engine = ForecastEngine.get_instance()
        self.simulator = ShockSimulator()
        self.graph_engine: Optional[GraphEngine] = None
        self.graph_enabled = settings.use_graph_propagation
        if self.graph_enabled:
            try:
                A = np.load(settings.adjacency_matrix_path)
                self.graph_engine = GraphEngine(
                    adjacency_matrix=A,
                    alpha=settings.graph_alpha,
                    iterations=settings.graph_iterations,
                )
                logger.info("ScenarioService: GraphEngine initialized")
            except Exception as e:
                logger.exception("ScenarioService: failed to init GraphEngine: %s", e)
                self.graph_enabled = False

    def run(
        self,
        sequence: np.ndarray,
        shock_params: Optional[ShockParams] = None,
    ) -> Dict[str, Any]:
        """Run the full orchestration and return a structured report.

        Args:
            sequence: input sequence array shape (seq_len, features)
            shock_params: optional ShockParams; if provided, a shocked scenario is created

        Returns:
            dict containing baseline, shocked (if any), impact, risk_analysis, recommendations, metadata
        """
        logger.info("ScenarioService.run: starting scenario run")

        # Baseline forecast
        baseline_result = self.engine.predict(sequence)
        baseline_preds_batch = baseline_result.get("predictions", [])
        if not baseline_preds_batch:
            raise RuntimeError("baseline model returned no predictions")
        baseline_preds = baseline_preds_batch[0]

        # Optionally apply spatial propagation to baseline predictions
        if self.graph_enabled and self.graph_engine is not None:
            # graph expects shape (horizon, num_roads) or (samples, num_roads)
            baseline_arr = np.array(baseline_preds)
            try:
                propagated_baseline = self.graph_engine.propagate_batch(baseline_arr.reshape(1, -1))
                baseline_preds = list(propagated_baseline.flatten())
            except Exception:
                logger.exception("ScenarioService: graph propagation failed for baseline")

        # Risk and decision for baseline
        baseline_risk = classify_risk_series(baseline_preds)
        baseline_decision = infrastructure_decision(max(baseline_preds))

        shocked_report = None
        report = None
        shock_applied = False

        if shock_params is not None:
            shock_applied = True
            # Apply shock to input sequence
            shocked_seq = self.simulator.apply_shock(sequence, shock_params)

            # Shocked forecast
            shocked_result = self.engine.predict(shocked_seq)
            shocked_preds_batch = shocked_result.get("predictions", [])
            if not shocked_preds_batch:
                raise RuntimeError("shocked model returned no predictions")
            shocked_preds = shocked_preds_batch[0]

            # optional graph propagation on shocked predictions
            if self.graph_enabled and self.graph_engine is not None:
                try:
                    propagated_shocked = self.graph_engine.propagate_batch(np.array(shocked_preds).reshape(1, -1))
                    shocked_preds = list(propagated_shocked.flatten())
                except Exception:
                    logger.exception("ScenarioService: graph propagation failed for shocked")

            # Risk and decision for shocked
            shocked_risk = classify_risk_series(shocked_preds)
            shocked_decision = infrastructure_decision(max(shocked_preds))

            # Compare using ShockSimulator utilities
            comparison: ComparisonResult = self.simulator.compare_scenarios(
                baseline_preds=np.array(baseline_preds),
                shocked_preds=np.array(shocked_preds),
                baseline_risk=baseline_risk,
                shocked_risk=shocked_risk,
            )

            report = self.simulator.generate_report(comparison)

            shocked_report = {
                "predictions": report["shocked"]["predictions"],
                "risk_level": report["shocked"]["risk_level"],
                "decision": shocked_decision,
            }

        # Build final response
        response: Dict[str, Any] = {
            "baseline": {"predictions": baseline_preds, "risk_level": baseline_risk, "decision": baseline_decision},
            "metadata": {
                "model_version": settings.model_version,
                "horizon": settings.horizon,
                "graph_enabled": bool(self.graph_enabled),
                "shock_applied": bool(shock_applied),
            },
        }

        if shocked_report is not None and report is not None:
            response.update({
                "shocked": shocked_report,
                "impact": report["impact"],
                "risk_analysis": report["risk_analysis"],
                "recommendations": report["recommendations"],
            })

        logger.info("ScenarioService.run: completed scenario run")
        return response