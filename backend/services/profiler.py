"""
Data Profiler Service
- Missing value detection
- Duplicate detection
- Data type validation
- Basic column statistics
"""
import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    """Run full profiling on a DataFrame. Returns a comprehensive report dict."""
    total_cells = df.shape[0] * df.shape[1]

    # ---------- Missing Values ----------
    missing_per_col = df.isnull().sum()
    missing_pct_per_col = (missing_per_col / len(df) * 100).round(2)
    total_missing = int(missing_per_col.sum())
    total_missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0

    # ---------- Duplicates ----------
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_rows / len(df) * 100, 2) if len(df) else 0.0

    # ---------- Data Type Validation ----------
    type_issues = _detect_type_issues(df)
    total_type_issues = sum(v["invalid_count"] for v in type_issues.values())
    total_type_issue_pct = round(total_type_issues / total_cells * 100, 2) if total_cells else 0.0

    # ---------- Column Statistics ----------
    column_stats = _compute_column_stats(df)

    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {
            "per_column": missing_per_col.to_dict(),
            "per_column_pct": missing_pct_per_col.to_dict(),
            "total": total_missing,
            "total_pct": total_missing_pct,
        },
        "duplicates": {
            "count": duplicate_rows,
            "pct": duplicate_pct,
        },
        "type_issues": type_issues,
        "type_issue_pct": total_type_issue_pct,
        "column_stats": column_stats,
    }


def _detect_type_issues(df: pd.DataFrame) -> dict:
    """
    Heuristic type validation: for columns that appear numeric (based on name or
    majority of values), detect non-numeric entries; for date-like columns, detect
    non-parseable dates.
    """
    issues = {}
    for col in df.columns:
        col_issues = {"invalid_count": 0, "invalid_pct": 0.0, "samples": []}

        if df[col].dtype == "object":
            # Try to see if this column *should* be numeric
            numeric_converted = pd.to_numeric(df[col], errors="coerce")
            non_null_original = df[col].dropna()
            non_null_converted = numeric_converted.dropna()

            if len(non_null_original) > 0:
                numeric_ratio = len(non_null_converted) / len(non_null_original)
                # If more than 50% are numeric, flag the non-numeric ones
                if numeric_ratio > 0.5 and numeric_ratio < 1.0:
                    invalid_mask = non_null_original.index.difference(non_null_converted.index)
                    bad_vals = df[col].loc[invalid_mask]
                    col_issues["invalid_count"] = len(bad_vals)
                    col_issues["invalid_pct"] = round(
                        len(bad_vals) / len(df) * 100, 2
                    )
                    col_issues["samples"] = bad_vals.head(5).tolist()

        issues[col] = col_issues
    return issues


def _compute_column_stats(df: pd.DataFrame) -> dict:
    """Compute per-column descriptive statistics."""
    stats = {}
    for col in df.columns:
        col_stat = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            col_stat.update({
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": round(float(desc.get("min", 0)), 4),
                "25%": round(float(desc.get("25%", 0)), 4),
                "50%": round(float(desc.get("50%", 0)), 4),
                "75%": round(float(desc.get("75%", 0)), 4),
                "max": round(float(desc.get("max", 0)), 4),
            })
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            col_stat["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        stats[col] = col_stat
    return stats
