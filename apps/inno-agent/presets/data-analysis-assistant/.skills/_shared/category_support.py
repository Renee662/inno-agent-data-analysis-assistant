"""Deterministic support and separation screening for categorical model terms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


CELL_COUNT_BENCHMARK = 5
EVENTS_PER_PARAMETER_BENCHMARK = 10.0


def screen_category_support(
    frame: pd.DataFrame,
    *,
    outcome: str,
    categorical_predictors: set[str],
    model_type: str,
    parameters_per_outcome_equation: int,
    design_matrix: pd.DataFrame | None = None,
    positive_class: str | None = None,
) -> dict[str, Any]:
    """Return evidence, not an automatic category-editing decision."""
    classification = model_type in {
        "logistic",
        "multinomial-logistic",
        "ordinal-logistic",
    }
    if not classification:
        return {
            "status": "not-applicable",
            "model_fitting_allowed": True,
            "model_type": model_type,
            "plain_explanation": "当前模型不属于本次分类结果支持度检查范围。",
            "factor_levels": [],
            "blocking_findings": [],
            "review_findings": [],
        }

    outcome_values = frame[outcome].astype(str)
    outcome_categories = sorted(outcome_values.unique().tolist())
    outcome_counts = {
        category: int((outcome_values == category).sum())
        for category in outcome_categories
    }
    denominator = max(int(parameters_per_outcome_equation), 1)
    events_per_parameter = {
        category: count / denominator for category, count in outcome_counts.items()
    }
    factor_levels: list[dict[str, Any]] = []
    blocking_findings: list[dict[str, Any]] = []
    review_findings: list[dict[str, Any]] = []

    for factor in sorted(categorical_predictors):
        factor_values = frame[factor].astype(str)
        for level in sorted(factor_values.unique().tolist()):
            mask = factor_values == level
            total = int(mask.sum())
            counts = {
                category: int(((outcome_values == category) & mask).sum())
                for category in outcome_categories
            }
            nonzero_categories = [category for category, count in counts.items() if count > 0]
            zero_categories = [category for category, count in counts.items() if count == 0]
            expected_counts = {
                category: total * outcome_counts[category] / max(len(frame), 1)
                for category in outcome_categories
            }
            minimum_observed = min(counts.values()) if counts else 0
            minimum_expected = min(expected_counts.values()) if expected_counts else 0.0
            risk_codes: list[str] = []
            blocking = False
            if model_type in {"logistic", "multinomial-logistic"} and zero_categories:
                risk_codes.append("separation-candidate")
                blocking = True
            elif model_type == "ordinal-logistic" and len(nonzero_categories) <= 1:
                risk_codes.append("separation-candidate")
                blocking = True
            if minimum_observed < CELL_COUNT_BENCHMARK or minimum_expected < CELL_COUNT_BENCHMARK:
                risk_codes.append("sparse-outcome-cell")
            item = {
                "factor": factor,
                "level": level,
                "total": total,
                "outcome_counts": counts,
                "minimum_observed_outcome_count": minimum_observed,
                "minimum_expected_outcome_count": minimum_expected,
                "zero_outcome_categories": zero_categories,
                "risk_codes": risk_codes,
                "blocking": blocking,
            }
            factor_levels.append(item)
            if blocking:
                blocking_findings.append(item)
            elif risk_codes:
                review_findings.append(item)

    low_information_outcomes = [
        {
            "outcome_category": category,
            "observations": outcome_counts[category],
            "parameters_per_outcome_equation": denominator,
            "observations_per_parameter": events_per_parameter[category],
        }
        for category in outcome_categories
        if events_per_parameter[category] < EVENTS_PER_PARAMETER_BENCHMARK
    ]
    review_findings.extend(low_information_outcomes)
    design_separation: dict[str, Any] = {"status": "not-checked"}
    if model_type == "logistic" and design_matrix is not None:
        if positive_class is None:
            raise ValueError("positive_class is required for binary design-separation screening")
        design = np.column_stack(
            [np.ones(len(design_matrix)), design_matrix.to_numpy(dtype=float)]
        )
        signs = np.where(outcome_values.to_numpy() == str(positive_class), 1.0, -1.0)
        parameter_columns = design.shape[1]
        objective = np.r_[np.zeros(parameter_columns * 2), -1.0]
        constraints = np.column_stack(
            [
                -(signs[:, None] * design),
                signs[:, None] * design,
                np.ones(len(design)),
            ]
        )
        l1_constraint = np.r_[np.ones(parameter_columns * 2), 0.0][None, :]
        optimization = linprog(
            objective,
            A_ub=np.vstack([constraints, l1_constraint]),
            b_ub=np.r_[np.zeros(len(design)), 1.0],
            bounds=[(0.0, None)] * (parameter_columns * 2) + [(None, None)],
            method="highs",
        )
        maximum_margin = (
            round(float(optimization.x[-1]), 12)
            if optimization.success and optimization.x is not None
            else None
        )
        separated = maximum_margin is not None and maximum_margin > 1e-8
        design_separation = {
            "status": "complete-separation" if separated else "not-detected",
            "maximum_margin_under_unit_l1_norm": maximum_margin,
            "design_terms": ["const", *map(str, design_matrix.columns)],
            "method": "linear-programming-maximum-margin",
        }
        if separated:
            blocking_findings.append(
                {
                    "reason": "multivariable-complete-separation",
                    "maximum_margin_under_unit_l1_norm": maximum_margin,
                    "explanation": (
                        "多个模型项组合后可以把正类和负类完全分开，普通Logistic最大似然系数不存在有限稳定解。"
                    ),
                }
            )
    status = (
        "revision-required"
        if blocking_findings
        else ("requires-review" if review_findings else "clear")
    )
    return {
        "status": status,
        "model_fitting_allowed": status == "clear",
        "model_type": model_type,
        "outcome_categories": outcome_categories,
        "outcome_counts": outcome_counts,
        "parameters_per_outcome_equation": denominator,
        "observations_per_parameter_by_outcome": events_per_parameter,
        "planning_benchmarks": {
            "minimum_cell_count": CELL_COUNT_BENCHMARK,
            "observations_per_parameter": EVENTS_PER_PARAMETER_BENCHMARK,
            "use_as_automatic_scientific_cutoff": False,
            "explanation": (
                "这些数值只用于提示近似推断可能不稳定；是否合并或排除类别必须结合类别含义、"
                "模型规模和研究目的由用户重新批准。零结果单元造成的分离风险属于单独的数学阻断。"
            ),
        },
        "factor_levels": factor_levels,
        "design_separation": design_separation,
        "blocking_findings": blocking_findings,
        "review_findings": review_findings,
        "plain_explanation": (
            "稀疏类别是某个类别样本或某种结果太少。分类分离是某个类别几乎或完全只对应一种结果，"
            "会让普通Logistic系数和优势比变得不稳定甚至无限大。"
        ),
    }


def flatten_category_support_rows(screen: dict[str, Any]) -> list[dict[str, Any]]:
    outcome_categories = [str(value) for value in screen.get("outcome_categories", [])]
    rows: list[dict[str, Any]] = []
    for item in screen.get("factor_levels", []):
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {
            "factor": item.get("factor"),
            "level": item.get("level"),
            "total": item.get("total"),
            "minimum_observed_outcome_count": item.get("minimum_observed_outcome_count"),
            "minimum_expected_outcome_count": item.get("minimum_expected_outcome_count"),
            "zero_outcome_categories": " | ".join(map(str, item.get("zero_outcome_categories", []))),
            "risk_codes": " | ".join(map(str, item.get("risk_codes", []))),
            "blocking": bool(item.get("blocking")),
        }
        counts = item.get("outcome_counts", {})
        for category in outcome_categories:
            row[f"outcome_count:{category}"] = int(counts.get(category, 0)) if isinstance(counts, dict) else 0
        rows.append(row)
    return rows
