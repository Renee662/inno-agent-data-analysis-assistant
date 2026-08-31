"""Deterministic, descriptive missingness-impact checks for selected variables."""

from __future__ import annotations

from typing import Any

import pandas as pd


NUMERIC_TYPES = {"numeric-continuous", "numeric-discrete"}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _categorical_distribution(series: pd.Series) -> list[dict[str, Any]]:
    observed = series.dropna()
    denominator = int(len(observed))
    counts = observed.value_counts(dropna=False, sort=False)
    return [
        {
            "value": _json_value(value),
            "count": int(count),
            "proportion": float(count / denominator) if denominator else None,
        }
        for value, count in counts.items()
    ]


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "median": None, "standard_deviation": None}
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "standard_deviation": float(numeric.std(ddof=1)) if len(numeric) > 1 else None,
    }


def _outcome_summary(series: pd.Series, outcome_type: str) -> dict[str, Any]:
    if outcome_type in NUMERIC_TYPES:
        return {"kind": "numeric", **_numeric_summary(series)}
    return {"kind": "categorical", "distribution": _categorical_distribution(series)}


def _max_category_gap(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left.get("kind") != "categorical" or right.get("kind") != "categorical":
        return None
    left_map = {str(row["value"]): row.get("proportion") for row in left["distribution"]}
    right_map = {str(row["value"]): row.get("proportion") for row in right["distribution"]}
    gaps = [
        abs(float(left_map.get(key) or 0.0) - float(right_map.get(key) or 0.0))
        for key in set(left_map) | set(right_map)
    ]
    return max(gaps) if gaps else None


def build_missingness_screen(
    frame: pd.DataFrame,
    variables: list[dict[str, str]],
) -> dict[str, Any]:
    """Compare observed and missing groups without claiming a missingness mechanism."""
    selected = [item["column"] for item in variables]
    missing_columns = [column for column in selected if bool(frame[column].isna().any())]
    outcome_item = next(item for item in variables if item["role"] == "outcome")
    outcome = outcome_item["column"]
    outcome_type = outcome_item.get("inferred_type", "unknown")
    complete_mask = frame[selected].notna().all(axis=1)
    retained = frame.loc[complete_mask, outcome]
    excluded = frame.loc[~complete_mask, outcome]
    retained_summary = _outcome_summary(retained, outcome_type)
    excluded_summary = _outcome_summary(excluded, outcome_type)

    by_field: list[dict[str, Any]] = []
    for item in variables:
        column = item["column"]
        mask = frame[column].isna()
        if not bool(mask.any()):
            continue
        missing_outcome = _outcome_summary(frame.loc[mask, outcome], outcome_type)
        observed_outcome = _outcome_summary(frame.loc[~mask, outcome], outcome_type)
        comparison: dict[str, Any] = {
            "column": column,
            "role": item["role"],
            "missing_rows": int(mask.sum()),
            "observed_rows": int((~mask).sum()),
            "missing_group_outcome": missing_outcome,
            "observed_group_outcome": observed_outcome,
        }
        gap = _max_category_gap(missing_outcome, observed_outcome)
        if gap is not None:
            comparison["max_outcome_proportion_gap"] = gap
        elif outcome_type in NUMERIC_TYPES:
            left = missing_outcome.get("mean")
            right = observed_outcome.get("mean")
            comparison["outcome_mean_difference"] = (
                float(left) - float(right) if left is not None and right is not None else None
            )
        by_field.append(comparison)

    any_missing = bool(missing_columns)
    return {
        "status": "review-required" if any_missing else "clear-no-selected-missingness",
        "diagnostic_scope": "descriptive-sample-composition",
        "selected_variables": selected,
        "input_rows": int(len(frame)),
        "complete_case_rows": int(complete_mask.sum()),
        "complete_case_excluded_rows": int((~complete_mask).sum()),
        "complete_case_excluded_rate": float((~complete_mask).mean()) if len(frame) else 0.0,
        "complete_case_outcome_retained": retained_summary,
        "complete_case_outcome_excluded": excluded_summary,
        "complete_case_max_outcome_proportion_gap": _max_category_gap(
            retained_summary, excluded_summary
        ),
        "field_comparisons": by_field,
        "interpretation": {
            "can_identify_mcar_mar_mnar": False,
            "can_prove_no_selection_bias": False,
            "model_estimate_sensitivity_included": False,
            "required_language": (
                "这些比较只能显示缺失与样本构成的关系，不能证明缺失是完全随机、随机或非随机。"
            ),
        },
    }


def flatten_missingness_screen(screen: dict[str, Any]) -> list[dict[str, Any]]:
    """Create a compact audit table for CSV/report use."""
    rows: list[dict[str, Any]] = []
    for item in screen.get("field_comparisons", []):
        rows.append(
            {
                "column": item.get("column"),
                "role": item.get("role"),
                "missing_rows": item.get("missing_rows"),
                "observed_rows": item.get("observed_rows"),
                "max_outcome_proportion_gap": item.get("max_outcome_proportion_gap"),
                "outcome_mean_difference": item.get("outcome_mean_difference"),
            }
        )
    return rows
