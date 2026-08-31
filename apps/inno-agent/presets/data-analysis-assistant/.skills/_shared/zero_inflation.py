"""Adjusted excess-zero screen for ordinary Poisson and negative-binomial models."""

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
        "plain_explanation": "过多零值检查只适用于Poisson和负二项计数模型。",
    }


def screen_zero_inflation(
    frame: pd.DataFrame,
    outcome: str,
    design_with_constant: pd.DataFrame,
    model_type: str,
    alpha: float,
) -> dict[str, Any]:
    if model_type not in {"poisson", "negative-binomial"}:
        raise ValueError("Excess-zero screening supports Poisson or negative-binomial models")
    if not 0 < alpha < 0.5:
        raise ValueError("Excess-zero alpha must be between 0 and 0.5")
    y = pd.to_numeric(frame[outcome], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError("Count outcome must contain finite non-negative integers")
    design = design_with_constant.astype(float)
    if len(y) <= design.shape[1] or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError("Excess-zero screening design is not identifiable")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_type == "poisson":
            result = sm.GLM(y, design, family=sm.families.Poisson()).fit(maxiter=300)
            fitted_mean = np.asarray(result.fittedvalues, dtype=float)
            fitted_alpha = None
            expected_zero_probability = np.exp(-fitted_mean)
        else:
            result = sm.NegativeBinomial(y, design).fit(disp=False, maxiter=300)
            fitted_mean = np.asarray(result.predict(design), dtype=float)
            fitted_alpha = float(np.asarray(result.params, dtype=float).reshape(-1)[-1])
            if not np.isfinite(fitted_alpha) or fitted_alpha <= 0:
                raise ValueError("Negative-binomial excess-zero screen produced invalid alpha")
            expected_zero_probability = np.power(
                1.0 + fitted_alpha * fitted_mean, -1.0 / fitted_alpha
            )
    critical = [
        warning
        for warning in caught
        if issubclass(warning.category, (ConvergenceWarning, HessianInversionWarning))
    ]
    if critical:
        raise ValueError("; ".join(str(warning.message) for warning in critical))
    if getattr(result, "converged", None) is False:
        raise ValueError("Excess-zero screening fit did not converge")
    if not np.isfinite(fitted_mean).all() or (fitted_mean <= 0).any():
        raise ValueError("Excess-zero screening produced invalid fitted means")
    if (
        not np.isfinite(expected_zero_probability).all()
        or (expected_zero_probability <= 0).any()
        or (expected_zero_probability >= 1).any()
    ):
        raise ValueError("Excess-zero screening produced invalid expected zero probabilities")
    observed_zero_count = int(np.sum(y == 0))
    expected_zero_count = float(np.sum(expected_zero_probability))
    variance = float(
        np.sum(expected_zero_probability * (1.0 - expected_zero_probability))
    )
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Excess-zero uncertainty is unavailable")
    statistic = float((observed_zero_count - expected_zero_count) / np.sqrt(variance))
    p_value = float(stats.norm.sf(statistic))
    detected = observed_zero_count > expected_zero_count and p_value < alpha
    return {
        "status": "excess-zeros-detected" if detected else "clear-no-detected-excess-zeros",
        "model_fitting_allowed": not detected,
        "test": "one-sided adjusted observed-versus-model-expected zero-count screen",
        "model_type": model_type,
        "null_hypothesis": "the fitted ordinary count model adequately explains the zero frequency",
        "rows": int(len(y)),
        "observed_zero_count": observed_zero_count,
        "observed_zero_rate": float(observed_zero_count / len(y)),
        "expected_zero_count": expected_zero_count,
        "expected_zero_rate": float(expected_zero_count / len(y)),
        "excess_zero_count": float(observed_zero_count - expected_zero_count),
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "negative_binomial_alpha": fitted_alpha,
        "decision": "excess-zeros-detected" if detected else "do-not-reject-ordinary-zero-frequency",
        "plain_explanation": (
            "该检查先按已批准因素拟合普通计数模型，再比较实际零值数与模型预计零值数。"
            "显著偏多表示普通Poisson或负二项模型可能没有充分描述零值产生机制。"
        ),
        "interpretation_boundary": (
            "过多零值可能来自结构性零值、遗漏异质性或均值结构错误；本检查不能单独确定原因，"
            "也不能自动选择零膨胀或门槛模型。"
        ),
        "required_action": (
            "use a specialized hurdle/zero-inflated workflow or revise the count-model structure"
            if detected
            else "ordinary count modeling may proceed with the recorded limitation"
        ),
    }


def materially_matches(approved: dict[str, Any], runtime: dict[str, Any]) -> bool:
    for key in ("status", "model_fitting_allowed", "test", "model_type", "decision"):
        if approved.get(key) != runtime.get(key):
            return False
    for key in (
        "observed_zero_count", "observed_zero_rate", "expected_zero_count",
        "expected_zero_rate", "excess_zero_count", "statistic", "p_value", "alpha",
    ):
        if not np.isclose(float(approved.get(key)), float(runtime.get(key)), rtol=1e-7, atol=1e-9):
            return False
    left_alpha = approved.get("negative_binomial_alpha")
    right_alpha = runtime.get("negative_binomial_alpha")
    if left_alpha is None or right_alpha is None:
        return left_alpha is None and right_alpha is None
    return bool(np.isclose(float(left_alpha), float(right_alpha), rtol=1e-7, atol=1e-9))
