"""Cluster-robust parallel-slopes screen for ordinal Logistic models."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def not_applicable() -> dict[str, Any]:
    return {
        "status": "not-applicable",
        "model_fitting_allowed": True,
        "plain_explanation": "比例优势假设只适用于有序Logistic模型。",
    }


def screen_proportional_odds(
    frame: pd.DataFrame,
    outcome: str,
    design_without_constant: pd.DataFrame,
    ordered_categories: list[str],
    alpha: float,
) -> dict[str, Any]:
    """Test equality of cumulative-logit slopes across outcome thresholds.

    The cumulative binary outcomes for one row are correlated, so the
    unrestricted stacked-binomial fit uses a row-clustered sandwich covariance.
    This is a Wald-style parallel-slopes screen, not an invalid likelihood-ratio
    comparison between non-nested ordinal and multinomial models.
    """
    if len(ordered_categories) < 3:
        raise ValueError("Proportional-odds screening requires at least three ordered categories")
    if not 0 < alpha < 0.5:
        raise ValueError("Proportional-odds alpha must be between 0 and 0.5")
    if design_without_constant.empty or design_without_constant.shape[1] < 1:
        raise ValueError("Proportional-odds screening requires at least one predictor term")
    values = frame[outcome].astype(str)
    if set(values.unique()) != set(ordered_categories):
        raise ValueError("Observed ordinal categories differ from the approved order")
    codes = pd.Series(
        pd.Categorical(values, categories=ordered_categories, ordered=True).codes,
        index=frame.index,
    )
    if (codes < 0).any():
        raise ValueError("Ordinal outcome contains an unrecognized category")
    design = design_without_constant.astype(float).reset_index(drop=True)
    threshold_count = len(ordered_categories) - 1
    row_count = len(design)
    long_outcome = np.concatenate(
        [(codes.to_numpy() <= threshold).astype(float) for threshold in range(threshold_count)]
    )
    threshold_ids = np.repeat(np.arange(threshold_count), row_count)
    cluster_ids = np.tile(np.arange(row_count), threshold_count)
    repeated_design = pd.concat([design] * threshold_count, ignore_index=True)
    threshold_dummies = pd.get_dummies(
        threshold_ids, prefix="threshold", drop_first=False, dtype=float
    )
    unrestricted_design = pd.concat([threshold_dummies, repeated_design], axis=1)
    deviation_columns: list[str] = []
    for threshold in range(1, threshold_count):
        mask = (threshold_ids == threshold).astype(float)
        for column in design.columns:
            name = f"deviation_threshold_{threshold}:{column}"
            unrestricted_design[name] = repeated_design[column].to_numpy() * mask
            deviation_columns.append(name)
    captured: list[warnings.WarningMessage]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unrestricted_result = sm.GLM(
            long_outcome,
            unrestricted_design.astype(float),
            family=sm.families.Binomial(),
        ).fit(
            maxiter=300,
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )
        captured = list(caught)
    critical = [
        warning
        for warning in captured
        if issubclass(warning.category, (ConvergenceWarning, HessianInversionWarning))
    ]
    if critical:
        raise ValueError("; ".join(str(warning.message) for warning in critical))
    if getattr(unrestricted_result, "converged", None) is False:
        raise ValueError("threshold-specific cumulative-logit screen did not converge")
    if not np.isfinite(np.asarray(unrestricted_result.params, dtype=float)).all():
        raise ValueError("threshold-specific cumulative-logit screen produced nonfinite parameters")

    deviation = unrestricted_result.params.loc[deviation_columns].to_numpy(dtype=float)
    covariance = unrestricted_result.cov_params().loc[
        deviation_columns, deviation_columns
    ].to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError("parallel-slopes screen produced a nonfinite covariance matrix")
    predictor_terms = int(design.shape[1])
    degrees_of_freedom = int((len(ordered_categories) - 2) * predictor_terms)
    statistic = float(deviation.T @ np.linalg.pinv(covariance) @ deviation)
    statistic = max(0.0, statistic)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    violation = p_value < alpha
    return {
        "status": "violation-detected" if violation else "clear-no-detected-violation",
        "model_fitting_allowed": not violation,
        "test": "cluster-robust Wald screen for equal cumulative-logit slopes",
        "null_hypothesis": "all predictor slopes are shared across cumulative outcome thresholds",
        "ordered_categories": ordered_categories,
        "predictor_terms": list(map(str, design.columns)),
        "statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "alpha": alpha,
        "decision": "reject-shared-slopes" if violation else "do-not-reject-shared-slopes",
        "plain_explanation": (
            "该检验把结局按各等级分界重新比较，并检查同一因素在不同分界上的系数是否一致。"
            "p值低于已批准显著性水平时，普通有序Logistic会把不同分界的影响过度合并。"
        ),
        "interpretation_boundary": (
            "未拒绝只表示当前样本未发现明显违背证据，不证明比例优势假设绝对成立；"
            "检验能力仍受样本量和类别稀疏程度影响。"
        ),
        "required_action": (
            "use multinomial logistic or a specialized partial-proportional-odds workflow"
            if violation
            else "ordinary ordinal logistic may proceed with the recorded limitation"
        ),
    }


def materially_matches(approved: dict[str, Any], runtime: dict[str, Any]) -> bool:
    keys = {
        "status",
        "model_fitting_allowed",
        "test",
        "ordered_categories",
        "predictor_terms",
        "degrees_of_freedom",
        "decision",
    }
    if any(approved.get(key) != runtime.get(key) for key in keys):
        return False
    for key in ("statistic", "p_value"):
        left = approved.get(key)
        right = runtime.get(key)
        if left is None or right is None:
            if left != right:
                return False
        elif not np.isclose(float(left), float(right), rtol=1e-7, atol=1e-9):
            return False
    return True
