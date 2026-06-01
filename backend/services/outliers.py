"""
Outlier Detection Service
- Isolation Forest
- IQR (Interquartile Range)
- Z-Score
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats


def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Run all three outlier detection methods on numeric columns.
    Returns a summary dict with per-column and aggregate results.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {
            "numeric_columns": [],
            "per_column": {},
            "total_outlier_pct": 0.0,
            "method_summary": {},
        }

    per_column = {}
    all_outlier_indices = set()

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 5:
            per_column[col] = {
                "iqr": {"count": 0, "indices": []},
                "zscore": {"count": 0, "indices": []},
                "isolation_forest": {"count": 0, "indices": []},
                "combined_count": 0,
            }
            continue

        iqr_result = _iqr_method(col_data)
        zscore_result = _zscore_method(col_data)
        iso_result = _isolation_forest_method(col_data)

        # Union of all outlier indices for this column
        combined = set(iqr_result["indices"]) | set(zscore_result["indices"]) | set(iso_result["indices"])
        all_outlier_indices.update(combined)

        per_column[col] = {
            "iqr": {"count": iqr_result["count"], "indices": iqr_result["indices"][:20]},
            "zscore": {"count": zscore_result["count"], "indices": zscore_result["indices"][:20]},
            "isolation_forest": {"count": iso_result["count"], "indices": iso_result["indices"][:20]},
            "combined_count": len(combined),
        }

    total_outlier_pct = round(len(all_outlier_indices) / len(df) * 100, 2) if len(df) else 0.0

    return {
        "numeric_columns": numeric_cols,
        "per_column": per_column,
        "total_outlier_pct": total_outlier_pct,
        "total_outlier_rows": len(all_outlier_indices),
    }


def _iqr_method(series: pd.Series) -> dict:
    """Detect outliers using the IQR method (1.5x IQR rule)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (series < lower) | (series > upper)
    indices = series[outlier_mask].index.tolist()
    return {"count": len(indices), "indices": indices}


def _zscore_method(series: pd.Series, threshold: float = 3.0) -> dict:
    """Detect outliers using Z-Score (default threshold = 3)."""
    z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
    outlier_mask = z_scores > threshold
    indices = series[outlier_mask].index.tolist()
    return {"count": len(indices), "indices": indices}


def _isolation_forest_method(series: pd.Series, contamination: float = 0.05) -> dict:
    """Detect outliers using Isolation Forest."""
    data = series.values.reshape(-1, 1)
    try:
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        predictions = iso_forest.fit_predict(data)
        outlier_mask = predictions == -1
        indices = series.index[outlier_mask].tolist()
        return {"count": len(indices), "indices": indices}
    except Exception:
        return {"count": 0, "indices": []}
