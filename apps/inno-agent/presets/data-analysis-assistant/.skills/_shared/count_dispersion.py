"""Deterministic overdispersion gate for ordinary Poisson regression."""

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
        "plain_explanation": "Poisson过度离散门禁只适用于普通Poisson回归。",
    }


def screen_count_dispersion(
    frame: pd.DataFrame,
    outcome: str,
    design_with_constant: pd.DataFrame,
    alpha: float,
) -> dict[str, Any]:
    """Run a one-sided Cameron-Trivedi-style NB2 auxiliary score screen."""
    if not 0 < alpha < 0.5:
        raise ValueError("Count-dispersion alpha must be between 0 and 0.5")
    y = pd.to_numeric(frame[outcome], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError("Poisson outcome must contain finite non-negative integers")
    design = design_with_constant.astype(float)
    if len(y) <= design.shape[1] or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError("Poisson screening design is not identifiable")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        poisson = sm.GLM(y, design, family=sm.families.Poisson()).fit(maxiter=300)
    critical = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if critical or getattr(poisson, "converged", None) is False:
        raise ValueError("Poisson screening fit did not converge")
    mu = np.asarray(poisson.fittedvalues, dtype=float)
    if not np.isfinite(mu).all() or (mu <= 0).any():
        raise ValueError("Poisson screening fit produced invalid fitted means")
    pearson_dispersion = float(np.sum((y - mu) ** 2 / mu) / poisson.df_resid)
    auxiliary_response = ((y - mu) ** 2 - y) / mu
    auxiliary = sm.OLS(auxiliary_response, mu[:, None]).fit(cov_type="HC3")
    coefficient = float(np.asarray(auxiliary.params)[0])
    standard_error = float(np.asarray(auxiliary.bse)[0])
    if not np.isfinite([coefficient, standard_error]).all() or standard_error <= 0:
        raise ValueError("Overdispersion screen produced invalid uncertainty")
    statistic = coefficient / standard_error
    p_value = float(stats.norm.sf(statistic))
    detected = coefficient > 0 and p_value < alpha
    return {
        "status": "overdispersion-detected" if detected else "clear-no-detected-overdispersion",
        "model_fitting_allowed": not detected,
        "test": "one-sided Cameron-Trivedi-style NB2 auxiliary score screen",
        "null_hypothesis": "conditional Poisson variance is adequate (NB2 alpha <= 0)",
        "auxiliary_coefficient": coefficient,
        "standard_error": standard_error,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "pearson_dispersion_descriptive": pearson_dispersion,
        "decision": "positive-overdispersion-detected" if detected else "do-not-reject-poisson-variance",
        "plain_explanation": (
            "该检查比较实际计数波动是否系统性超过Poisson在相同预测均值下允许的波动。"
            "显著的正向结果表示普通Poisson可能把不确定性写得过小。"
        ),
        "interpretation_boundary": (
            "Pearson离散度只作描述，不使用固定经验阈值自动换模；正式门禁依据已批准显著性水平的单侧检验。"
            "未拒绝不证明Poisson方差假设绝对成立。"
        ),
        "required_action": (
            "return to model choice and seek approval for negative-binomial regression or stop"
            if detected
            else "ordinary Poisson may proceed with the recorded limitation"
        ),
    }


def materially_matches(approved: dict[str, Any], runtime: dict[str, Any]) -> bool:
    for key in ("status", "model_fitting_allowed", "test", "decision"):
        if approved.get(key) != runtime.get(key):
            return False
    for key in (
        "auxiliary_coefficient",
        "standard_error",
        "statistic",
        "p_value",
        "alpha",
        "pearson_dispersion_descriptive",
    ):
        if not np.isclose(float(approved.get(key)), float(runtime.get(key)), rtol=1e-7, atol=1e-9):
            return False
    return True
