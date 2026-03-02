"""Risk classification utilities.

Provides richer classification based on a series of predictions using
max, mean, and percentile thresholds.
"""

from typing import Sequence
import numpy as np


def classify_risk_value(value: float) -> str:
	"""Classify a single congestion value into risk buckets.

	Thresholds:
	  - Critical: >= 0.75
	  - Emerging: >= 0.55
	  - Stable: otherwise
	"""
	try:
		v = float(value)
	except Exception:
		return "Unknown"

	if v >= 0.75:
		return "Critical"
	if v >= 0.55:
		return "Emerging"
	return "Stable"


def classify_risk_series(predictions: Sequence[float]) -> str:
	"""Classify risk for a series of predictions using max, mean, and percentile.

	Logic (aggregated):
	  - compute max_val, mean_val, p90
	  - escalate to Critical if any of: max_val >= 0.8, p90 >= 0.75
	  - Emerging if mean_val >= 0.55 or max_val >= 0.7
	  - Stable otherwise
	"""
	arr = np.asarray(list(predictions), dtype=float)
	if arr.size == 0:
		return "Unknown"

	max_v = float(np.max(arr))
	mean_v = float(np.mean(arr))
	p90 = float(np.percentile(arr, 90))

	if max_v >= 0.8 or p90 >= 0.75:
		return "Critical"
	if mean_v >= 0.55 or max_v >= 0.7:
		return "Emerging"
	return "Stable"

