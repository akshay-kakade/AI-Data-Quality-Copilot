"""
Data Drift Detection Service
Compare a baseline (production) dataset against a target (new) dataset.
Detects:
  - Mean changes (numeric)
  - Distribution changes (KS test for numeric, chi-squared for categorical)
  - Missing value changes
  - Category distribution changes
"""
import pandas as pd
import numpy as np
from scipy import stats


def detect_drift(baseline_df: pd.DataFrame, target_df: pd.DataFrame) -> dict:
    """
    Compare baseline and target DataFrames for data drift.
    Only compares columns present in both datasets.
    """
    common_cols = list(set(baseline_df.columns) & set(target_df.columns))
    if not common_cols:
        return {
            "common_columns": [],
            "drifted_columns": [],
            "drift_summary": {},
            "overall_drift_score": 100.0,
        }

    drift_details = {}
    drifted_columns = []

    for col in common_cols:
        col_drift = _analyze_column_drift(baseline_df[col], target_df[col], col)
        drift_details[col] = col_drift
        if col_drift["is_drifted"]:
            drifted_columns.append(col)

    # Overall drift score: 100 = no drift (perfect), 0 = everything drifted
    drift_ratio = len(drifted_columns) / len(common_cols) if common_cols else 0
    overall_drift_score = round((1 - drift_ratio) * 100, 2)

    return {
        "common_columns": common_cols,
        "drifted_columns": drifted_columns,
        "drift_summary": drift_details,
        "overall_drift_score": overall_drift_score,
        "total_columns_compared": len(common_cols),
        "total_drifted": len(drifted_columns),
    }


def _analyze_column_drift(baseline_col: pd.Series, target_col: pd.Series, col_name: str) -> dict:
    """Analyze drift for a single column."""
    result = {
        "column": col_name,
        "is_drifted": False,
        "drift_type": None,
        "details": {},
    }

    # ---------- Missing Value Changes ----------
    base_missing_pct = round(baseline_col.isnull().mean() * 100, 2)
    target_missing_pct = round(target_col.isnull().mean() * 100, 2)
    missing_change = round(target_missing_pct - base_missing_pct, 2)
    result["details"]["missing_value_change"] = {
        "baseline_pct": base_missing_pct,
        "target_pct": target_missing_pct,
        "change": missing_change,
    }

    is_numeric_base = pd.api.types.is_numeric_dtype(baseline_col)
    is_numeric_target = pd.api.types.is_numeric_dtype(target_col)

    if is_numeric_base and is_numeric_target:
        result["drift_type"] = "numeric"

        # Mean change
        base_mean = round(float(baseline_col.mean()), 4) if not baseline_col.dropna().empty else 0
        target_mean = round(float(target_col.mean()), 4) if not target_col.dropna().empty else 0
        mean_change = round(target_mean - base_mean, 4)
        mean_change_pct = round(
            abs(mean_change / base_mean * 100) if base_mean != 0 else 0, 2
        )

        result["details"]["mean_change"] = {
            "baseline_mean": base_mean,
            "target_mean": target_mean,
            "absolute_change": mean_change,
            "pct_change": mean_change_pct,
        }

        # KS Test for distribution change
        base_clean = baseline_col.dropna()
        target_clean = target_col.dropna()
        if len(base_clean) > 1 and len(target_clean) > 1:
            ks_stat, ks_pvalue = stats.ks_2samp(base_clean, target_clean)
            result["details"]["distribution_test"] = {
                "test": "Kolmogorov-Smirnov",
                "statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_pvalue), 6),
                "significant": ks_pvalue < 0.05,
            }
            if ks_pvalue < 0.05:
                result["is_drifted"] = True

    else:
        result["drift_type"] = "categorical"

        # Category distribution changes
        base_dist = baseline_col.dropna().value_counts(normalize=True)
        target_dist = target_col.dropna().value_counts(normalize=True)

        all_categories = set(base_dist.index) | set(target_dist.index)
        new_categories = set(target_dist.index) - set(base_dist.index)
        removed_categories = set(base_dist.index) - set(target_dist.index)

        result["details"]["category_changes"] = {
            "baseline_unique": len(base_dist),
            "target_unique": len(target_dist),
            "new_categories": list(new_categories)[:10],
            "removed_categories": list(removed_categories)[:10],
        }

        # Chi-squared-like comparison using PSI (Population Stability Index)
        psi = _calculate_psi(base_dist, target_dist, all_categories)
        result["details"]["stability_index"] = {
            "psi": round(psi, 4),
            "interpretation": "Significant drift" if psi > 0.2 else (
                "Moderate drift" if psi > 0.1 else "No significant drift"
            ),
        }
        if psi > 0.2:
            result["is_drifted"] = True

    # Flag large missing value changes as drift
    if abs(missing_change) > 10:
        result["is_drifted"] = True

    return result


def _calculate_psi(base_dist: pd.Series, target_dist: pd.Series, all_categories: set) -> float:
    """Calculate Population Stability Index (PSI) for categorical distributions."""
    eps = 1e-6
    psi = 0.0
    for cat in all_categories:
        base_prop = base_dist.get(cat, eps)
        target_prop = target_dist.get(cat, eps)
        if base_prop == 0:
            base_prop = eps
        if target_prop == 0:
            target_prop = eps
        psi += (target_prop - base_prop) * np.log(target_prop / base_prop)
    return abs(psi)
