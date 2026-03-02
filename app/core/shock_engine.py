"""Scenario simulation engine for traffic shock analysis.

Simulates traffic disruptions (vehicle count spikes, peak hour multipliers)
and compares baseline vs shocked forecasts to assess impact on congestion.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RiskShift(str, Enum):
    """Risk classification change from baseline to shocked scenario."""
    ESCALATED = "escalated"
    STABLE = "stable"
    MITIGATED = "mitigated"


@dataclass
class ShockParams:
    """Parameters defining a traffic shock scenario."""
    vehicle_count_increase: float = 0.0  # percentage increase in vehicles (e.g., 0.2 = +20%)
    peak_hour_multiplier: float = 1.0  # multiplier during peak hours (e.g., 1.5 = 50% increase)
    affected_time_steps: Optional[List[int]] = None  # which horizon steps to apply shock; None = all


@dataclass
class ComparisonResult:
    """Result of baseline vs shocked scenario comparison."""
    baseline_predictions: List[float]
    shocked_predictions: List[float]
    delta_congestion: List[float]  # shocked - baseline per horizon step
    delta_percentage: List[float]  # (shocked - baseline) / baseline * 100, clamped to avoid inf
    max_delta: float  # largest absolute change
    avg_delta_percentage: float  # average % change across horizon
    risk_shift: RiskShift  # risk classification change
    baseline_risk: str  # risk level for baseline
    shocked_risk: str  # risk level for shocked
    severity_score: float  # 0-1 scale: how much does shock degrade conditions


class ShockSimulator:
    """Simulates traffic shocks and compares forecasts."""

    def __init__(self, congestion_floor: float = 0.0, congestion_ceiling: float = 1.0):
        """Initialize shock simulator.

        Args:
            congestion_floor: Minimum valid congestion value.
            congestion_ceiling: Maximum valid congestion value.
        """
        self.congestion_floor = congestion_floor
        self.congestion_ceiling = congestion_ceiling

    def apply_shock(self, sequence: np.ndarray, shock: ShockParams) -> np.ndarray:
        """Apply traffic shock to an input sequence.

        Shocks are applied by scaling the input sequence features (vehicle counts).
        This amplifies congestion without directly modifying target predictions.

        Args:
            sequence: 2D array of shape (seq_len, num_features). Assumed first feature is vehicle count.
            shock: ShockParams defining the magnitude and timing of the shock.

        Returns:
            Shocked sequence with vehicle count increased per shock parameters.
        """
        shocked = sequence.copy()

        # Apply vehicle count increase to all timesteps
        if shock.vehicle_count_increase > 0:
            shocked[:, 0] *= (1 + shock.vehicle_count_increase)

        # Apply peak hour multiplier to affected timesteps
        if shock.peak_hour_multiplier != 1.0:
            if shock.affected_time_steps is None:
                affected_steps = range(len(shocked))
            else:
                affected_steps = shock.affected_time_steps

            for t in affected_steps:
                if 0 <= t < len(shocked):
                    shocked[t, 0] *= shock.peak_hour_multiplier

        return shocked

    def compare_scenarios(
        self,
        baseline_preds: np.ndarray,
        shocked_preds: np.ndarray,
        baseline_risk: str,
        shocked_risk: str,
    ) -> ComparisonResult:
        """Compare baseline and shocked scenario predictions.

        Args:
            baseline_preds: 1D array of baseline predictions for each horizon step.
            shocked_preds: 1D array of shocked predictions for each horizon step.
            baseline_risk: Risk level string for baseline ("Critical", "Emerging", "Stable").
            shocked_risk: Risk level string for shocked scenario.

        Returns:
            ComparisonResult with detailed comparison metrics.
        """
        baseline_preds = np.asarray(baseline_preds).flatten()
        shocked_preds = np.asarray(shocked_preds).flatten()

        if len(baseline_preds) != len(shocked_preds):
            raise ValueError(
                f"prediction arrays must have same length; got {len(baseline_preds)} vs {len(shocked_preds)}"
            )

        # Calculate deltas
        delta_congestion = shocked_preds - baseline_preds

        # Calculate percentage change, clamped to avoid division by zero
        delta_percentage = []
        for base, shock in zip(baseline_preds, shocked_preds):
            if base > 1e-6:
                pct = ((shock - base) / base) * 100
            else:
                pct = 0.0 if shock <= 1e-6 else 100.0
            delta_percentage.append(pct)

        delta_percentage = np.array(delta_percentage)

        # Determine risk shift
        risk_shift = self._classify_risk_shift(baseline_risk, shocked_risk)

        # Calculate severity score using combined metrics (percentages are in %)
        abs_delta_pct = np.abs(delta_percentage)
        max_delta_pct = float(np.max(abs_delta_pct)) / 100.0
        avg_delta_pct = float(np.mean(abs_delta_pct)) / 100.0
        # duration above threshold (10% by default)
        duration_threshold = 10.0
        duration_above = float(np.sum(abs_delta_pct > duration_threshold))
        duration_ratio = duration_above / float(len(abs_delta_pct)) if len(abs_delta_pct) > 0 else 0.0

        severity = (
            0.4 * max_delta_pct + 0.4 * avg_delta_pct + 0.2 * duration_ratio
        )

        max_delta = float(np.max(np.abs(delta_congestion)))
        avg_pct = float(np.mean(delta_percentage))

        return ComparisonResult(
            baseline_predictions=baseline_preds.tolist(),
            shocked_predictions=shocked_preds.tolist(),
            delta_congestion=delta_congestion.tolist(),
            delta_percentage=delta_percentage.tolist(),
            max_delta=max_delta,
            avg_delta_percentage=avg_pct,
            risk_shift=risk_shift,
            baseline_risk=baseline_risk,
            shocked_risk=shocked_risk,
            severity_score=severity,
        )

    def _classify_risk_shift(self, baseline_risk: str, shocked_risk: str) -> RiskShift:
        """Classify how risk has changed from baseline to shocked.

        Args:
            baseline_risk: Risk level for baseline.
            shocked_risk: Risk level for shocked scenario.

        Returns:
            RiskShift enum indicating escalation, stability, or mitigation.
        """
        risk_hierarchy = {"Stable": 0, "Emerging": 1, "Critical": 2}

        base_level = risk_hierarchy.get(baseline_risk, 1)
        shock_level = risk_hierarchy.get(shocked_risk, 1)

        if shock_level > base_level:
            return RiskShift.ESCALATED
        elif shock_level < base_level:
            return RiskShift.MITIGATED
        else:
            return RiskShift.STABLE

    def generate_report(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Generate a structured scenario comparison report.

        Args:
            comparison: ComparisonResult from compare_scenarios().

        Returns:
            Dictionary with formatted report for display/logging.
        """
        return {
            "scenario_type": "traffic_shock",
            "baseline": {
                "predictions": [round(p, 4) for p in comparison.baseline_predictions],
                "risk_level": comparison.baseline_risk,
            },
            "shocked": {
                "predictions": [round(p, 4) for p in comparison.shocked_predictions],
                "risk_level": comparison.shocked_risk,
            },
            "impact": {
                "delta_congestion": [round(d, 4) for d in comparison.delta_congestion],
                "delta_percentage": [round(p, 2) for p in comparison.delta_percentage],
                "max_delta": round(comparison.max_delta, 4),
                "avg_delta_percentage": round(comparison.avg_delta_percentage, 2),
                "severity_score": round(comparison.severity_score, 3),
            },
            "risk_analysis": {
                "baseline_risk": comparison.baseline_risk,
                "shocked_risk": comparison.shocked_risk,
                "risk_shift": comparison.risk_shift.value,
            },
            "recommendations": self._generate_recommendations(comparison),
        }

    def _generate_recommendations(self, comparison: ComparisonResult) -> List[str]:
        """Generate actionable recommendations based on scenario results.

        Args:
            comparison: ComparisonResult from compare_scenarios().

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if comparison.risk_shift == RiskShift.ESCALATED:
            recommendations.append("Risk escalation detected. Increase monitoring and response readiness.")
            if comparison.severity_score > 0.7:
                recommendations.append(
                    "High-severity shock. Pre-emptive infrastructure or traffic diversion recommended."
                )

        elif comparison.risk_shift == RiskShift.MITIGATED:
            recommendations.append("Shock impact is mitigated. Current measures appear effective.")

        if comparison.avg_delta_percentage > 50:
            recommendations.append("Significant congestion increase (>50% avg). Prepare intervention protocols.")

        if max(comparison.shocked_predictions) > 0.85:
            recommendations.append("Near-capacity conditions forecast. Activate demand management.")

        if not recommendations:
            recommendations.append("Scenario within acceptable variance. No urgent action required.")

        return recommendations
