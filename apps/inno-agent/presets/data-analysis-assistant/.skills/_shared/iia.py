"""Deterministic Hausman-McFadden sensitivity screen for multinomial Logistic models."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning


def not_applicable() -> dict[str, Any]:
    return {
        "status": "not-applicable",
        "model_fitting_allowed": True,
        "plain_explanation": "IIA检查只适用于无序多分类Logistic模型。",
    }


def _fit_multinomial(codes: pd.Series, design: pd.DataFrame) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.MNLogit(codes.astype(int), design.astype(float)).fit(
            method="newton", maxiter=300, disp=False
        )
    critical = [
        warning
        for warning in caught
        if issubclass(warning.category, (ConvergenceWarning, HessianInversionWarning))
    ]
    if critical:
        raise ValueError("; ".join(str(warning.message) for warning in critical))
    if getattr(result, "mle_retvals", {}).get("converged") is not True:
        raise ValueError("multinomial sensitivity fit did not converge")
    parameters = np.asarray(result.params, dtype=float)
    covariance = np.asarray(result.cov_params(), dtype=float)
    if not np.isfinite(parameters).all() or not np.isfinite(covariance).all():
        raise ValueError("multinomial sensitivity fit produced nonfinite estimates")
    return result


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[int(original_index)]))
        running = max(running, candidate)
        adjusted[int(original_index)] = running
    return adjusted.tolist()


def screen_iia(
    frame: pd.DataFrame,
    outcome: str,
    design_with_constant: pd.DataFrame,
    outcome_categories: list[str],
    alpha: float,
) -> dict[str, Any]:
    """Refit after deleting each non-reference alternative and compare common coefficients.

    This is a generalized Hausman-McFadden sensitivity screen.  A positive-eigenvalue
    generalized inverse is used because the classical covariance difference can be
    indefinite in finite samples; that fact is retained in every deletion record.
    """
    if len(outcome_categories) < 3:
        raise ValueError("IIA screening requires at least three unordered categories")
    if not 0 < alpha < 0.5:
        raise ValueError("IIA alpha must be between 0 and 0.5")
    if design_with_constant.empty or design_with_constant.shape[1] < 2:
        raise ValueError("IIA screening requires an intercept and at least one predictor term")
    values = frame[outcome].astype(str).reset_index(drop=True)
    categories = [str(value) for value in outcome_categories]
    if set(values.unique()) != set(categories):
        raise ValueError("Observed outcome categories differ from the approved multinomial categories")
    design = design_with_constant.astype(float).reset_index(drop=True)
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError("IIA screening design is rank deficient")
    full_codes = pd.Series(pd.Categorical(values, categories=categories).codes)
    if (full_codes < 0).any():
        raise ValueError("Multinomial outcome contains an unrecognized category")
    full_result = _fit_multinomial(full_codes, design)
    full_parameters = np.asarray(full_result.params, dtype=float)
    full_covariance = np.asarray(full_result.cov_params(), dtype=float)
    term_count = design.shape[1]
    if full_parameters.shape != (term_count, len(categories) - 1):
        raise ValueError("Full multinomial parameter layout is unexpected")

    deletion_tests: list[dict[str, Any]] = []
    valid_test_positions: list[int] = []
    raw_p_values: list[float] = []
    for removed in categories[1:]:
        retained = [category for category in categories if category != removed]
        keep = values != removed
        reduced_values = values.loc[keep].reset_index(drop=True)
        reduced_design = design.loc[keep].reset_index(drop=True)
        reduced_codes = pd.Series(
            pd.Categorical(reduced_values, categories=retained).codes
        )
        record: dict[str, Any] = {
            "removed_category": removed,
            "retained_categories": retained,
            "rows_retained": int(keep.sum()),
            "status": "not-evaluated",
        }
        try:
            reduced_result = _fit_multinomial(reduced_codes, reduced_design)
            reduced_parameters = np.asarray(reduced_result.params, dtype=float)
            reduced_covariance = np.asarray(reduced_result.cov_params(), dtype=float)
            common_full_blocks = [categories.index(category) - 1 for category in retained[1:]]
            full_indices = np.concatenate(
                [
                    np.arange(block * term_count, (block + 1) * term_count)
                    for block in common_full_blocks
                ]
            )
            full_common = full_parameters[:, common_full_blocks].reshape(-1, order="F")
            reduced_vector = reduced_parameters.reshape(-1, order="F")
            if full_common.shape != reduced_vector.shape:
                raise ValueError("common coefficient layouts differ after category deletion")
            full_common_covariance = full_covariance[np.ix_(full_indices, full_indices)]
            covariance_difference = (
                reduced_covariance - full_common_covariance
            )
            covariance_difference = (covariance_difference + covariance_difference.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_difference)
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            tolerance = scale * 1e-8
            positive = eigenvalues > tolerance
            rank = int(np.sum(positive))
            if rank < 1:
                raise ValueError("Hausman covariance difference has no positive information direction")
            difference = reduced_vector - full_common
            projection = eigenvectors[:, positive].T @ difference
            statistic = float(np.sum((projection**2) / eigenvalues[positive]))
            statistic = max(0.0, statistic)
            p_value = float(stats.chi2.sf(statistic, rank))
            record.update(
                {
                    "status": "evaluated",
                    "statistic": statistic,
                    "degrees_of_freedom": rank,
                    "p_value_raw": p_value,
                    "negative_covariance_directions": int(np.sum(eigenvalues < -tolerance)),
                    "zero_covariance_directions": int(np.sum(np.abs(eigenvalues) <= tolerance)),
                    "maximum_absolute_coefficient_change": float(np.max(np.abs(difference))),
                }
            )
            valid_test_positions.append(len(deletion_tests))
            raw_p_values.append(p_value)
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        deletion_tests.append(record)

    if not valid_test_positions:
        return {
            "status": "not-evaluated",
            "model_fitting_allowed": False,
            "test": "generalized Hausman-McFadden category-deletion sensitivity screen",
            "reference_category": categories[0],
            "outcome_categories": categories,
            "predictor_terms": list(map(str, design.columns)),
            "alpha": alpha,
            "deletion_tests": deletion_tests,
            "plain_explanation": "IIA敏感性检查未能得到可解释的类别删除比较；这不能视为IIA成立。",
            "required_action": "use a specialized multinomial choice workflow or revise the outcome structure",
        }

    adjusted = _holm_adjust(raw_p_values)
    for position, adjusted_p in zip(valid_test_positions, adjusted):
        deletion_tests[position]["p_value_holm"] = adjusted_p
        deletion_tests[position]["decision"] = (
            "sensitivity-detected" if adjusted_p < alpha else "no-detected-sensitivity"
        )
    violation = any(value < alpha for value in adjusted)
    incomplete = len(valid_test_positions) != len(deletion_tests)
    return {
        "status": "sensitivity-detected" if violation else "clear-no-detected-sensitivity",
        "model_fitting_allowed": not violation,
        "test": "generalized Hausman-McFadden category-deletion sensitivity screen",
        "null_hypothesis": "common category log-odds coefficients remain stable when another alternative is removed",
        "reference_category": categories[0],
        "outcome_categories": categories,
        "predictor_terms": list(map(str, design.columns)),
        "alpha": alpha,
        "multiplicity_adjustment": "Holm family-wise adjustment across category deletions",
        "evaluated_deletions": len(valid_test_positions),
        "total_deletions": len(deletion_tests),
        "incomplete": incomplete,
        "minimum_adjusted_p_value": float(min(adjusted)),
        "decision": "reject-choice-set-stability" if violation else "do-not-reject-choice-set-stability",
        "deletion_tests": deletion_tests,
        "plain_explanation": (
            "该检查依次移除一个非参照类别，并比较其余类别相对参照类别的系数是否明显改变。"
            "校正后p值低于已批准判断水平时，普通多分类Logistic的IIA假设存在不稳定证据。"
        ),
        "interpretation_boundary": (
            "未发现敏感性不等于证明IIA成立；Hausman-McFadden检查会受样本量、稀疏类别和有限样本协方差影响，"
            "还应结合类别之间是否存在天然相似或嵌套关系判断。"
        ),
        "required_action": (
            "use a specialized nested/multinomial-probit choice workflow or revise substantively compatible categories"
            if violation
            else "ordinary multinomial logistic may proceed with the recorded limitation"
        ),
    }


def materially_matches(approved: dict[str, Any], runtime: dict[str, Any]) -> bool:
    structural_keys = {
        "status",
        "model_fitting_allowed",
        "test",
        "reference_category",
        "outcome_categories",
        "predictor_terms",
        "multiplicity_adjustment",
        "evaluated_deletions",
        "total_deletions",
        "incomplete",
        "decision",
    }
    if any(approved.get(key) != runtime.get(key) for key in structural_keys):
        return False
    left_tests = approved.get("deletion_tests")
    right_tests = runtime.get("deletion_tests")
    if not isinstance(left_tests, list) or not isinstance(right_tests, list):
        return False
    if len(left_tests) != len(right_tests):
        return False
    numeric_keys = {
        "statistic",
        "p_value_raw",
        "p_value_holm",
        "maximum_absolute_coefficient_change",
    }
    for left, right in zip(left_tests, right_tests):
        for key in set(left) | set(right):
            if key in numeric_keys and left.get(key) is not None and right.get(key) is not None:
                if not np.isclose(float(left[key]), float(right[key]), rtol=1e-7, atol=1e-9):
                    return False
            elif left.get(key) != right.get(key):
                return False
    return True
