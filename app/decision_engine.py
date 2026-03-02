"""Policy decision rules for infrastructure interventions."""


def infrastructure_decision(congestion_value: float) -> str:
	"""Return a high-level recommendation based on predicted congestion.

	This simple rule set is used for the MVP; thresholds are tunable via config.
	"""
	if congestion_value > 0.75:
		return "Immediate Infrastructure Upgrade Recommended"
	elif congestion_value > 0.55:
		return "Operational Optimization Required"
	else:
		return "System Stable"

