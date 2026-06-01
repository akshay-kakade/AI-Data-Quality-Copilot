"""
Quality Score Calculator
Fixed formula with no user-configurable weights.

Quality Score =
  100
  - Missing Value Penalty
  - Duplicate Penalty
  - Outlier Penalty
  - Invalid Type Penalty
  - Drift Penalty

Weights:
  Missing Values  = 35%
  Duplicates      = 20%
  Outliers        = 15%
  Invalid Types   = 15%
  Data Drift      = 15%

Each sub-score is 0-100 where 100 = no issues.
Final score is the weighted average of all sub-scores.
"""

# Fixed weights
WEIGHTS = {
    "missing": 0.35,
    "duplicates": 0.20,
    "outliers": 0.15,
    "invalid_types": 0.15,
    "drift": 0.15,
}


def calculate_quality_score(
    missing_pct: float,
    duplicate_pct: float,
    outlier_pct: float,
    invalid_type_pct: float,
    drift_score: float = 100.0,
) -> dict:
    """
    Calculate the composite quality score.

    Args:
        missing_pct: Percentage of total cells that are missing (0-100)
        duplicate_pct: Percentage of rows that are duplicates (0-100)
        outlier_pct: Percentage of rows with outliers (0-100)
        invalid_type_pct: Percentage of cells with type issues (0-100)
        drift_score: Drift score from drift detection (0-100, 100 = no drift).
                     Defaults to 100 when no drift analysis is performed.

    Returns:
        Dict with quality_score, risk_level, and breakdown.
    """
    # Convert percentages to sub-scores (100 = perfect, 0 = worst)
    missing_score = max(0, 100 - missing_pct)
    duplicate_score = max(0, 100 - duplicate_pct)
    outlier_score = max(0, 100 - outlier_pct)
    type_score = max(0, 100 - invalid_type_pct)

    # Weighted average
    quality_score = (
        missing_score * WEIGHTS["missing"]
        + duplicate_score * WEIGHTS["duplicates"]
        + outlier_score * WEIGHTS["outliers"]
        + type_score * WEIGHTS["invalid_types"]
        + drift_score * WEIGHTS["drift"]
    )

    quality_score = round(quality_score, 2)
    quality_score = max(0, min(100, quality_score))

    risk_level = _get_risk_level(quality_score)

    return {
        "quality_score": quality_score,
        "risk_level": risk_level,
        "breakdown": {
            "missing_score": round(missing_score, 2),
            "duplicate_score": round(duplicate_score, 2),
            "outlier_score": round(outlier_score, 2),
            "type_score": round(type_score, 2),
            "drift_score": round(drift_score, 2),
        },
        "weights": WEIGHTS,
    }


def _get_risk_level(score: float) -> str:
    """Map quality score to a human-readable risk level."""
    if score >= 90:
        return "Low"
    elif score >= 75:
        return "Medium"
    elif score >= 50:
        return "High"
    else:
        return "Critical"
