#!/usr/bin/env python3
"""Fit an approved statistical model and create diagnostics and plots."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings as python_warnings
from datetime import datetime
from pathlib import Path
from typing import Any

_conversation_root = os.environ.get("INNO_CONVERSATION_DIR")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(
        (Path(_conversation_root) if _conversation_root else Path.cwd())
        / "work"
        / "matplotlib-cache"
    ),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as statsmodels_package
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationWarning,
    SingularMatrixWarning,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from category_support import screen_category_support  # noqa: E402
from file_utils import sha256_file  # noqa: E402
from model_design import apply_continuous_forms  # noqa: E402
from model_registry import SUPPORTED_MODEL_TYPE_SET  # noqa: E402
from count_dispersion import (  # noqa: E402
    materially_matches as count_dispersion_materially_matches,
    not_applicable as count_dispersion_not_applicable,
    screen_count_dispersion,
)
from iia import (  # noqa: E402
    materially_matches as iia_materially_matches,
    not_applicable as iia_not_applicable,
    screen_iia,
)
from negative_binomial_need import (  # noqa: E402
    materially_matches as negative_binomial_need_materially_matches,
    not_applicable as negative_binomial_need_not_applicable,
    screen_negative_binomial_need,
)
from proportional_odds import materially_matches, not_applicable, screen_proportional_odds  # noqa: E402
from zero_inflation import (  # noqa: E402
    materially_matches as zero_inflation_materially_matches,
    not_applicable as zero_inflation_not_applicable,
    screen_zero_inflation,
)


PRIMARY_BLUE_GRAY = "#4F6D7A"
SECONDARY_BLUE_GRAY = "#8CA4AF"
DARK_BLUE_GRAY = "#2F3E46"
LIGHT_GRID = "#D9E3E8"
MIN_CATEGORY_OBSERVATIONS = 5
MAX_CLASSIFICATION_COEFFICIENT = 25.0
MAX_CLASSIFICATION_STD_ERROR = 1_000.0
CHINESE_FONT_FALLBACKS = [
    "Microsoft YaHei",
    "Noto Sans SC",
    "DengXian",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
FACTOR_OMNIBUS_COLUMNS = (
    "factor",
    "reference_category",
    "test",
    "statistic",
    "degrees_of_freedom",
    "p_value",
    "p_value_adjusted_bh",
)
CONTINUOUS_SHAPE_COLUMNS = (
    "variable",
    "form",
    "overall_test",
    "overall_statistic",
    "overall_degrees_of_freedom",
    "overall_p_value",
    "overall_p_value_adjusted_bh",
    "nonlinear_terms",
    "nonlinear_test",
    "nonlinear_statistic",
    "nonlinear_degrees_of_freedom",
    "nonlinear_p_value",
    "nonlinear_p_value_adjusted_bh",
)
INFLUENCE_DIAGNOSTIC_COLUMNS = (
    "cleaned_data_row",
    "candidate_reasons",
    "leverage",
    "standardized_residual",
    "cook_distance",
    "priority_score",
    "case_deletion_evaluated",
    "refit_status",
    "max_standardized_parameter_change",
    "sign_flip_count",
    "significance_flip_count",
    "refit_error",
)


def configure_academic_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": CHINESE_FONT_FALLBACKS,
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.edgecolor": DARK_BLUE_GRAY,
            "axes.labelcolor": DARK_BLUE_GRAY,
            "text.color": DARK_BLUE_GRAY,
            "xtick.color": DARK_BLUE_GRAY,
            "ytick.color": DARK_BLUE_GRAY,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        }
    )


def polish_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK_BLUE_GRAY)
    ax.spines["bottom"].set_color(DARK_BLUE_GRAY)
    ax.tick_params(colors=DARK_BLUE_GRAY, labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            color=LIGHT_GRID,
            linewidth=0.7,
            alpha=0.55,
        )


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit an approved model and write audited outputs.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--preparation-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Cannot read {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object")
    return value


def scalar(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    return str(value)


def encode_categorical_predictors(
    frame: pd.DataFrame,
    categorical: set[str],
    references: dict[str, str],
) -> pd.DataFrame:
    encoded = frame.copy()
    if set(references) != categorical:
        missing = sorted(categorical - set(references))
        extra = sorted(set(references) - categorical)
        fail(
            "Approved categorical reference map does not match categorical predictors; "
            f"missing={missing}, extra={extra}"
        )
    for column in sorted(categorical):
        values = encoded[column].astype(str)
        observed = sorted(values.unique().tolist())
        reference = str(references[column])
        if reference not in observed:
            fail(
                f"Approved reference category {reference!r} is not observed for "
                f"{column!r}: {observed}"
            )
        categories = [reference, *(value for value in observed if value != reference)]
        encoded[column] = pd.Categorical(values, categories=categories, ordered=False)
    design = pd.get_dummies(
        encoded, columns=sorted(categorical), drop_first=True, dtype=float
    )
    for column in sorted(categorical):
        forbidden = f"{column}_{references[column]}"
        if forbidden in design.columns:
            fail(f"Reference category was not omitted from the design matrix: {forbidden}")
    return design


def validate_continuous_form_specifications(
    specifications: Any,
    continuous_columns: set[str],
    model_type: str,
) -> dict[str, dict[str, Any]]:
    if model_type not in SUPPORTED_MODEL_TYPE_SET:
        if specifications not in ({}, None):
            fail("Continuous functional forms are unsupported for this model type")
        return {}
    if not isinstance(specifications, dict) or set(map(str, specifications)) != continuous_columns:
        fail(
            f"Approved {model_type.upper()} specification must contain one functional form for every "
            "continuous predictor/control"
        )
    validated: dict[str, dict[str, Any]] = {}
    for column in sorted(continuous_columns):
        item = specifications.get(column)
        if not isinstance(item, dict) or item.get("form") not in {
            "linear", "quadratic", "restricted-cubic-spline"
        }:
            fail(f"Invalid approved functional form for {column!r}")
        expected_terms = {
            "linear": [column],
            "quadratic": [column, f"{column}__quadratic"],
            "restricted-cubic-spline": [column, f"{column}__rcs1", f"{column}__rcs2"],
        }[str(item["form"])]
        if item.get("term_names") != expected_terms:
            fail(f"Approved transformed-term map is inconsistent for {column!r}")
        if item.get("selection_source") not in {
            "explicit-approved-domain-override",
            "approved-task-metadata",
            "automatic-fixed-flexible-default",
            "automatic-limited-support-linear",
        } or not str(item.get("selection_rationale") or "").strip():
            fail(f"Approved functional-form decision provenance is incomplete for {column!r}")
        center = float(item.get("center"))
        scale = float(item.get("scale"))
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            fail(f"Approved scaling values are invalid for {column!r}")
        if item["form"] == "restricted-cubic-spline":
            knots = item.get("knots_standardized")
            if (
                not isinstance(knots, list)
                or len(knots) != 4
                or not np.isfinite(np.asarray(knots, dtype=float)).all()
                or any(float(left) >= float(right) for left, right in zip(knots, knots[1:]))
            ):
                fail(f"Approved spline knots are invalid for {column!r}")
        validated[column] = item
    return validated


def joint_wald_test(
    result: Any, design_columns: list[str], terms: list[str], label: str
) -> tuple[float, int, float]:
    missing = [term for term in terms if term not in design_columns]
    if missing:
        fail(f"Terms are missing from the fitted design for {label}: {missing}")
    indices = [design_columns.index(term) for term in terms]
    parameters = np.asarray(result.params, dtype=float).reshape(-1)[indices]
    covariance = np.asarray(result.cov_params(), dtype=float)[np.ix_(indices, indices)]
    degrees_of_freedom = int(np.linalg.matrix_rank(covariance))
    if degrees_of_freedom <= 0:
        fail(f"Joint Wald covariance is singular for {label}")
    statistic = float(parameters.T @ np.linalg.pinv(covariance) @ parameters)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        fail(f"Joint Wald test is nonfinite for {label}")
    return statistic, degrees_of_freedom, p_value


def safe_figure_stem(value: str) -> str:
    stem = "".join(character if character.isalnum() else "-" for character in value).strip("-")
    return stem or "continuous-predictor"


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        fail("Multiplicity correction received an invalid p-value")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return [float(value) for value in adjusted]


def categorical_factor_for_term(
    term: Any, categorical_references: dict[str, str]
) -> str | None:
    term_text = str(term)
    for column in sorted(categorical_references, key=len, reverse=True):
        if term_text.startswith(f"{column}_"):
            return column
    return None


def categorical_omnibus_tests(
    result: Any,
    model_type: str,
    design_columns: list[str],
    categorical_references: dict[str, str],
) -> list[dict[str, Any]]:
    if not categorical_references:
        return []
    covariance = np.asarray(result.cov_params(), dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        fail("Model covariance matrix is unavailable for categorical omnibus tests")
    if model_type == "multinomial-logistic":
        parameter_matrix = np.asarray(result.params, dtype=float)
        if parameter_matrix.ndim != 2:
            fail("Multinomial parameters do not have the expected matrix shape")
        parameter_vector = parameter_matrix.reshape(-1, order="F")
        category_blocks = parameter_matrix.shape[1]
    else:
        parameter_vector = np.asarray(result.params, dtype=float).reshape(-1)
        category_blocks = 1
    if covariance.shape[0] != len(parameter_vector):
        fail("Parameter and covariance dimensions differ for categorical omnibus tests")

    tests: list[dict[str, Any]] = []
    for factor, reference in sorted(categorical_references.items()):
        base_indices = [
            index
            for index, term in enumerate(design_columns)
            if term.startswith(f"{factor}_")
        ]
        if not base_indices:
            fail(f"No encoded coefficients were found for categorical factor {factor!r}")
        indices = [
            block * len(design_columns) + index
            for block in range(category_blocks)
            for index in base_indices
        ]
        beta = parameter_vector[indices]
        covariance_subset = covariance[np.ix_(indices, indices)]
        degrees_of_freedom = int(np.linalg.matrix_rank(covariance_subset))
        if degrees_of_freedom <= 0:
            fail(f"Categorical omnibus covariance is singular for {factor!r}")
        statistic = float(beta.T @ np.linalg.pinv(covariance_subset) @ beta)
        p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            fail(f"Categorical omnibus test is nonfinite for {factor!r}")
        tests.append(
            {
                "factor": factor,
                "reference_category": reference,
                "test": "joint-wald-chi-square",
                "statistic": statistic,
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": p_value,
            }
        )
    adjusted = benjamini_hochberg([float(item["p_value"]) for item in tests])
    for item, adjusted_p in zip(tests, adjusted, strict=True):
        item["p_value_adjusted_bh"] = adjusted_p
    return tests


def auc_rank(y: np.ndarray, score: np.ndarray) -> float | None:
    positive = y == 1
    n1 = int(positive.sum())
    n0 = int((~positive).sum())
    if not n1 or not n0:
        return None
    ranks = stats.rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary_roc_points(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    ordered = y_true[order].astype(int)
    positives = max(int(ordered.sum()), 1)
    negatives = max(int(len(ordered) - ordered.sum()), 1)
    true_positive = np.r_[0, np.cumsum(ordered)] / positives
    false_positive = np.r_[0, np.cumsum(1 - ordered)] / negatives
    return false_positive, true_positive


def predictive_cross_validation(
    model_type: str,
    design: pd.DataFrame,
    observed: pd.Series,
    outcome_categories: list[str],
    goal: str,
    folds: int = 5,
) -> dict[str, Any]:
    if goal != "prediction":
        return {
            "status": "not-applicable",
            "model_performance_claim_allowed": False,
            "reason": "the approved task goal is not prediction",
            "interpretation_boundary": "样本内拟合指标只能描述当前数据，不能表述为新数据预测能力。",
        }
    n = len(design)
    if n < folds * 10:
        return {
            "status": "not-evaluated",
            "model_performance_claim_allowed": False,
            "reason": f"fewer than {folds * 10} rows are available for {folds}-fold validation",
        }
    classification = model_type in {"logistic", "multinomial-logistic", "ordinal-logistic"}
    rng = np.random.default_rng(20260822)
    fold_ids = np.empty(n, dtype=int)
    if classification:
        values = np.asarray(observed, dtype=int)
        for category in np.unique(values):
            positions = np.flatnonzero(values == category)
            if len(positions) < folds:
                return {
                    "status": "not-evaluated",
                    "model_performance_claim_allowed": False,
                    "reason": f"outcome category {category} has fewer than {folds} rows",
                }
            shuffled = rng.permutation(positions)
            fold_ids[shuffled] = np.arange(len(shuffled)) % folds
    else:
        shuffled = rng.permutation(n)
        fold_ids[shuffled] = np.arange(n) % folds

    numeric_observed = np.asarray(observed, dtype=float)
    scalar_predictions = np.full(n, np.nan, dtype=float)
    probability_predictions = (
        np.full((n, len(outcome_categories)), np.nan, dtype=float)
        if model_type in {"multinomial-logistic", "ordinal-logistic"}
        else None
    )
    fold_records: list[dict[str, Any]] = []
    try:
        for fold in range(folds):
            test_mask = fold_ids == fold
            train_mask = ~test_mask
            x_train = design.loc[train_mask].reset_index(drop=True)
            x_test = design.loc[test_mask].reset_index(drop=True)
            y_train = pd.Series(numeric_observed[train_mask])
            if model_type == "ols":
                fold_result = sm.OLS(y_train, x_train).fit()
                prediction = np.asarray(fold_result.predict(x_test), dtype=float)
            elif model_type == "logistic":
                fold_result = sm.Logit(y_train.astype(int), x_train).fit(
                    method="newton", maxiter=200, disp=False
                )
                prediction = np.asarray(fold_result.predict(x_test), dtype=float)
            elif model_type == "poisson":
                fold_result = sm.GLM(
                    y_train, x_train, family=sm.families.Poisson()
                ).fit(maxiter=300)
                prediction = np.asarray(fold_result.predict(x_test), dtype=float)
            elif model_type == "negative-binomial":
                fold_result = sm.NegativeBinomial(y_train, x_train).fit(
                    method="newton", maxiter=300, disp=False
                )
                prediction = np.asarray(fold_result.predict(x_test), dtype=float)
            elif model_type == "multinomial-logistic":
                fold_result = sm.MNLogit(y_train.astype(int), x_train).fit(
                    method="newton", maxiter=300, disp=False
                )
                prediction = np.asarray(fold_result.predict(x_test), dtype=float)
            elif model_type == "ordinal-logistic":
                fold_model = OrderedModel(y_train.astype(int), x_train, distr="logit")
                fold_result = fold_model.fit(method="bfgs", maxiter=300, disp=False)
                prediction = np.asarray(
                    fold_result.model.predict(fold_result.params, exog=x_test), dtype=float
                )
            else:
                raise ValueError(f"Unsupported predictive validation model: {model_type}")
            if not np.isfinite(prediction).all():
                raise ValueError(f"fold {fold + 1} produced nonfinite predictions")
            if probability_predictions is not None:
                if prediction.shape != (int(test_mask.sum()), len(outcome_categories)):
                    raise ValueError(f"fold {fold + 1} probability columns differ from the approved outcome categories")
                probability_predictions[test_mask, :] = prediction
            else:
                scalar_predictions[test_mask] = prediction.reshape(-1)
            fold_records.append(
                {
                    "fold": fold + 1,
                    "training_rows": int(train_mask.sum()),
                    "validation_rows": int(test_mask.sum()),
                }
            )
    except Exception as exc:
        return {
            "status": "not-evaluated",
            "model_performance_claim_allowed": False,
            "method": f"deterministic stratified {folds}-fold cross-validation",
            "reason": f"{type(exc).__name__}: {exc}",
            "completed_folds": len(fold_records),
        }

    metrics: dict[str, Any]
    if probability_predictions is not None:
        probabilities = np.clip(probability_predictions, 1e-12, 1.0)
        predicted_codes = np.argmax(probabilities, axis=1)
        true_codes = numeric_observed.astype(int)
        metrics = {
            "cross_validated_accuracy": float(np.mean(predicted_codes == true_codes)),
            "cross_validated_log_loss": float(
                -np.mean(np.log(probabilities[np.arange(n), true_codes]))
            ),
            "cross_validated_ordinal_mean_absolute_category_error": (
                float(np.mean(np.abs(predicted_codes - true_codes)))
                if model_type == "ordinal-logistic"
                else None
            ),
        }
    elif model_type == "logistic":
        probabilities = np.clip(scalar_predictions, 1e-12, 1 - 1e-12)
        truth = numeric_observed.astype(int)
        metrics = {
            "cross_validated_accuracy_at_0_5": float(
                np.mean((probabilities >= 0.5).astype(int) == truth)
            ),
            "cross_validated_log_loss": float(
                -np.mean(truth * np.log(probabilities) + (1 - truth) * np.log(1 - probabilities))
            ),
            "cross_validated_brier_score": float(np.mean((truth - probabilities) ** 2)),
            "cross_validated_roc_auc": auc_rank(truth, probabilities),
        }
    else:
        errors = numeric_observed - scalar_predictions
        metrics = {
            "cross_validated_rmse": float(np.sqrt(np.mean(errors**2))),
            "cross_validated_mae": float(np.mean(np.abs(errors))),
            "cross_validated_r_squared": (
                float(1.0 - np.sum(errors**2) / np.sum((numeric_observed - numeric_observed.mean()) ** 2))
                if model_type == "ols" and np.var(numeric_observed) > 0
                else None
            ),
        }
    return {
        "status": "completed",
        "model_performance_claim_allowed": True,
        "method": f"deterministic {'stratified ' if classification else ''}{folds}-fold cross-validation",
        "folds": folds,
        "random_seed": 20260822,
        "rows": n,
        "preprocessing_scope": (
            "the approved variable set, reference categories, and functional-form structure were locked before validation; "
            "no outcome-guided model tuning was performed across folds"
        ),
        "metrics": metrics,
        "fold_records": fold_records,
        "interpretation_boundary": "这些指标来自折外预测，可用于描述当前数据条件下的内部验证表现，但不能替代外部数据验证。",
        "_out_of_fold_scalar_predictions": (
            scalar_predictions.tolist() if probability_predictions is None else None
        ),
        "_out_of_fold_probability_predictions": (
            probability_predictions.tolist() if probability_predictions is not None else None
        ),
    }


def calibration_table(
    observed: np.ndarray, predicted: np.ndarray, groups: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": observed, "predicted": predicted}).dropna()
    unique = int(frame["predicted"].nunique())
    if unique < 2:
        return pd.DataFrame(columns=["predicted", "observed", "count"])
    frame["group"] = pd.qcut(
        frame["predicted"], q=min(groups, unique), duplicates="drop"
    )
    return frame.groupby("group", observed=False).agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        count=("observed", "size"),
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def write_json_artifacts(output_dir: Path, artifacts: dict[str, Any]) -> None:
    """Write named JSON artifacts with the workflow's stable text format."""
    for filename, payload in artifacts.items():
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def write_analysis_tables(
    output_dir: Path,
    results: pd.DataFrame,
    diagnostics: list[dict[str, Any]],
    factor_tests: list[dict[str, Any]],
    continuous_shape_tests: list[dict[str, Any]],
    influence_rows: list[dict[str, Any]],
) -> None:
    """Write analysis CSV artifacts with their stable schemas and encoding."""
    results.to_csv(output_dir / "model-results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(
        output_dir / "model-diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(factor_tests, columns=FACTOR_OMNIBUS_COLUMNS).to_csv(
        output_dir / "factor-omnibus-tests.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(continuous_shape_tests, columns=CONTINUOUS_SHAPE_COLUMNS).to_csv(
        output_dir / "continuous-shape-tests.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(influence_rows, columns=INFLUENCE_DIAGNOSTIC_COLUMNS).to_csv(
        output_dir / "influence-diagnostics.csv", index=False, encoding="utf-8-sig"
    )


def build_learning_prompts(
    relationship_prompt: dict[str, Any],
    diagnostic_prompt: dict[str, Any],
    collinearity_summary: dict[str, Any],
    severe_vif_terms: list[str],
    influence_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the ordered post-model learning prompts from computed evidence."""
    prompt_sequence = ["observation", "diagnostic-reasoning"]
    collinearity_prompt: dict[str, Any] | None = None
    if collinearity_summary["status"] == "severe":
        prompt_sequence.insert(0, "collinearity-review")
        collinearity_prompt = {
            "question": "检测到严重共线性后，本次结果应如何处理？",
            "plain_explanation": collinearity_summary["plain_explanation"],
            "options": [
                "返回模型设置，调整重复变量后重新拟合",
                "保留当前模型，但不把受影响单项系数解释为稳定独立作用",
                "我还不理解，请先简要解释",
            ],
            "evidence_answer": "需要用户选择；系统不会自动删除变量。",
            "evidence_feedback": (
                f"本次最大VIF={collinearity_summary['maximum_vif']:.3f}，"
                f"有{len(severe_vif_terms)}个模型项达到VIF≥10。"
            ),
        }

    influence_prompt: dict[str, Any] | None = None
    if influence_summary["status"] == "not-evaluated" or influence_summary.get(
        "candidate_count", 0
    ):
        prompt_sequence.insert(
            1 if collinearity_prompt is not None else 0, "influence-review"
        )
        if influence_summary["status"] == "not-evaluated":
            question = "本模型的高影响记录诊断没有成功，应如何处理？"
            options = [
                "保留模型，但明确写明该项未评估并限制结论",
                "停止并改用能够完成影响诊断的专门流程",
                "我还不理解，请先简要解释",
            ]
            feedback = f"诊断失败原因：{influence_summary.get('reason', '未记录')}。这不等于没有高影响记录。"
        else:
            question = f"发现 {influence_summary.get('candidate_count', 0)} 条可能明显影响模型的记录，应如何处理？"
            options = [
                "保留原模型，并把逐条删一敏感性结果纳入解释",
                "返回数据核查阶段，检查候选记录是否为录入或测量错误",
                "我还不理解，请先简要解释",
            ]
            feedback = (
                f"系统只对优先级最高的 {influence_summary.get('case_deletion_evaluated_count', 0)} 条逐条重拟合，"
                f"没有自动删除记录；最大标准化参数变化为 {scalar(influence_summary.get('maximum_standardized_parameter_change'))}。"
            )
        influence_prompt = {
            "question": question,
            "plain_explanation": "异常值关注数据看起来是否特殊；高影响记录关注删掉它后模型结论是否明显变化。候选标记只是复核提示，不能作为自动删除依据。",
            "options": options,
            "evidence_answer": "需要用户选择；系统不会自动删除记录。",
            "evidence_feedback": feedback,
        }

    learning_prompts = {
        "status": "ready",
        "sequence": prompt_sequence,
        "observation": relationship_prompt,
        "diagnostic_reasoning": diagnostic_prompt,
        "instruction": (
            "严格按 sequence 逐项展示并等待用户选择。若存在 collinearity-review，"
            "修改变量的选择必须返回模型设置；保留选择也不能解除单项系数解释限制。"
            "若存在 influence-review，先用一两句解释异常值与高影响记录的区别，再展示当前模型证据。"
        ),
    }
    if collinearity_prompt is not None:
        learning_prompts["collinearity_review"] = collinearity_prompt
    if influence_prompt is not None:
        learning_prompts["influence_review"] = influence_prompt
    return learning_prompts


def workspace_relative_path(path: Path) -> str | None:
    """Return a browser-safe path relative to the active Inno workspace."""
    workspace = os.environ.get("INNO_WORKSPACE_DIR", "").strip()
    if not workspace:
        return None
    try:
        return path.resolve().relative_to(Path(workspace).resolve()).as_posix()
    except ValueError:
        return None


def attach_workspace_image(
    prompt: dict[str, Any], image_path: Path, alt: str, caption: str
) -> None:
    """Attach browser-safe workspace image metadata when the image is in scope."""
    workspace_path = workspace_relative_path(image_path)
    if workspace_path:
        prompt["workspace_image_path"] = workspace_path
        prompt["image_alt"] = alt
        prompt["image_caption"] = caption


def outcome_codes(
    values: pd.Series, categories: list[str], label: str
) -> pd.Series:
    observed = values.astype(str)
    unknown = sorted(set(observed.unique()) - set(categories))
    if unknown:
        fail(f"{label} categories no longer match the approved specification: {unknown}")
    categorical = pd.Categorical(observed, categories=categories, ordered=True)
    codes = pd.Series(categorical.codes, index=values.index)
    if (codes < 0).any():
        fail(f"{label} contains missing or unrecognized categories")
    return codes


def validate_sparse_predictor_levels(
    frame: pd.DataFrame,
    categorical_predictors: set[str],
) -> None:
    """Stop before fitting when a categorical predictor level is too sparse."""
    for column in sorted(categorical_predictors):
        counts = frame[column].astype(str).value_counts(dropna=False)
        sparse = counts[counts < MIN_CATEGORY_OBSERVATIONS]
        if not sparse.empty:
            details = ", ".join(f"{level}={int(count)}" for level, count in sparse.items())
            fail(
                f"Sparse predictor category in {column!r}: {details}. "
                f"Each retained level requires at least {MIN_CATEGORY_OBSERVATIONS} complete observations."
            )


def validate_outcome_category_support(
    values: pd.Series,
    model_type: str,
    design_columns: int,
) -> None:
    if model_type not in {"logistic", "multinomial-logistic", "ordinal-logistic"}:
        return
    counts = values.astype(str).value_counts(dropna=False)
    minimum = MIN_CATEGORY_OBSERVATIONS
    if model_type in {"logistic", "multinomial-logistic"}:
        minimum = max(minimum, design_columns + 1)
    sparse = counts[counts < minimum]
    if not sparse.empty:
        details = ", ".join(f"{level}={int(count)}" for level, count in sparse.items())
        fail(
            f"Sparse outcome category: {details}. This {model_type} specification "
            f"requires at least {minimum} complete observations in every outcome category."
        )


def validate_observed_separation(
    design_without_constant: pd.DataFrame,
    outcome_codes_series: pd.Series,
    model_type: str,
) -> None:
    """Catch deterministic categorical splits before iterative fitting."""
    if model_type not in {"logistic", "multinomial-logistic", "ordinal-logistic"}:
        return
    outcome_levels = sorted(pd.Series(outcome_codes_series).unique().tolist())
    for column in design_without_constant.columns:
        values = design_without_constant[column]
        unique_values = pd.Series(values).dropna().unique()
        if len(unique_values) != 2:
            continue
        table = pd.crosstab(values, outcome_codes_series)
        table = table.reindex(columns=outcome_levels, fill_value=0)
        if model_type in {"logistic", "multinomial-logistic"} and (table == 0).any(axis=None):
            fail(
                f"Complete or quasi-complete separation detected for predictor term {column!r}; "
                "at least one predictor level contains no observations from an outcome category."
            )
        if model_type == "ordinal-logistic" and (table > 0).sum(axis=1).min() < 2:
            fail(
                f"Complete separation detected for ordinal predictor term {column!r}; "
                "a predictor level maps to only one outcome category."
            )


def validate_count_separation(
    design_without_constant: pd.DataFrame,
    outcome: pd.Series,
    model_type: str,
) -> None:
    """Catch binary groups with only structural zero counts."""
    if model_type not in {"poisson", "negative-binomial"}:
        return
    if float(outcome.sum()) <= 0:
        fail(f"{model_type} outcome contains no positive counts")
    for column in design_without_constant.columns:
        values = design_without_constant[column]
        if pd.Series(values).dropna().nunique() != 2:
            continue
        grouped_sums = pd.Series(outcome).groupby(values).sum()
        if (grouped_sums <= 0).any():
            fail(
                f"Count-model separation detected for predictor term {column!r}; "
                "one predictor level contains no positive outcome counts."
            )


def validate_fitted_model(
    result: Any,
    model_type: str,
    fitted_probabilities: np.ndarray | None,
) -> None:
    """Refuse nonconvergent, singular, nonfinite, or separated fitted models."""
    if model_type != "ols":
        converged = getattr(result, "converged", None)
        if converged is False:
            fail("Model did not converge; results were not written.")
        mle_retvals = getattr(result, "mle_retvals", None)
        if isinstance(mle_retvals, dict) and mle_retvals.get("converged") is not True:
            fail(f"Model did not converge: {mle_retvals}")

    arrays = {
        "coefficient estimates": np.asarray(result.params, dtype=float),
        "standard errors": np.asarray(result.bse, dtype=float),
        "p-values": np.asarray(result.pvalues, dtype=float),
        "confidence intervals": np.asarray(result.conf_int(), dtype=float),
    }
    for label, values in arrays.items():
        if values.size == 0 or not np.isfinite(values).all():
            fail(f"Model produced nonfinite {label}; results were not written.")
    if (arrays["standard errors"] <= 0).any():
        fail("Model produced zero or negative standard errors; the fitted system is unidentified.")

    try:
        covariance = np.asarray(result.cov_params(), dtype=float)
    except Exception as exc:
        fail(f"Model covariance matrix is unavailable: {type(exc).__name__}: {exc}")
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not np.isfinite(covariance).all()
        or np.linalg.matrix_rank(covariance) < covariance.shape[0]
    ):
        fail("Model covariance matrix is singular or nonfinite; results were not written.")

    if model_type in {"logistic", "multinomial-logistic", "ordinal-logistic"}:
        coefficients = arrays["coefficient estimates"]
        standard_errors = arrays["standard errors"]
        if np.max(np.abs(coefficients)) > MAX_CLASSIFICATION_COEFFICIENT:
            fail(
                "Extreme classification coefficient detected; complete or quasi-complete "
                "separation is likely, so results were not written."
            )
        if np.max(standard_errors) > MAX_CLASSIFICATION_STD_ERROR:
            fail(
                "Extreme classification standard error detected; the model is not stably "
                "identified, so results were not written."
            )
        if fitted_probabilities is not None:
            if not np.isfinite(fitted_probabilities).all():
                fail("Model produced nonfinite fitted probabilities.")
            if (fitted_probabilities < 0).any() or (fitted_probabilities > 1).any():
                fail("Model produced invalid fitted probabilities outside [0, 1].")


def influence_screen(
    model_type: str,
    result: Any,
    design: pd.DataFrame,
    observed: np.ndarray,
    fitted: np.ndarray,
    observed_class_codes: np.ndarray | None = None,
    fitted_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a model-aware candidate screen without treating failure as zero."""
    n = len(design)
    parameter_count = max(int(design.shape[1]), 1)
    if n == 0:
        raise ValueError("Influence diagnostics require at least one fitted row")
    matrix = np.asarray(design, dtype=float)
    fitted = np.asarray(fitted, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if len(fitted) != n or len(observed) != n:
        raise ValueError("Influence inputs do not match the fitted sample")

    cook: np.ndarray | None = None
    if model_type == "ols":
        weights = np.ones(n, dtype=float)
        variance = np.full(
            n, max(float(getattr(result, "mse_resid", np.nan)), np.finfo(float).eps)
        )
        method = "OLS杠杆值、标准化残差和Cook距离"
    elif model_type == "logistic":
        weights = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        variance = weights.copy()
        method = "Logistic加权杠杆值、标准化Pearson残差和近似Cook距离"
    elif model_type == "poisson":
        weights = np.clip(fitted, 1e-10, None)
        variance = weights.copy()
        method = "Poisson加权杠杆值、标准化Pearson残差和近似Cook距离"
    elif model_type == "negative-binomial":
        alpha = float(np.asarray(result.params, dtype=float).reshape(-1)[-1])
        variance = np.clip(fitted + max(alpha, 0.0) * fitted**2, 1e-10, None)
        weights = np.clip(fitted**2 / variance, 1e-10, None)
        method = "负二项加权杠杆值、标准化Pearson残差和近似Cook距离"
    elif model_type in {"multinomial-logistic", "ordinal-logistic"}:
        weights = np.ones(n, dtype=float)
        variance = np.ones(n, dtype=float)
        method = (
            "使用设计杠杆值和实际类别偏差惊异度确定逐条删一优先级；"
            "本模型不声称具有Cook距离"
        )
    else:
        raise ValueError(f"Unsupported influence model: {model_type}")

    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    information_inverse = np.linalg.pinv(weighted_matrix.T @ weighted_matrix)
    leverage = np.einsum(
        "ij,jk,ik->i", weighted_matrix, information_inverse, weighted_matrix
    )
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-10)

    if model_type == "multinomial-logistic":
        if fitted_probabilities is None or observed_class_codes is None:
            raise ValueError("Classification influence screen requires probabilities and observed codes")
        probabilities = np.asarray(fitted_probabilities, dtype=float)
        codes = np.asarray(observed_class_codes, dtype=int)
        if probabilities.shape[0] != n or len(codes) != n:
            raise ValueError("Classification probability rows do not match the fitted sample")
        observed_probability = np.clip(probabilities[np.arange(n), codes], 1e-12, 1.0)
        standardized_residual = np.sqrt(-2.0 * np.log(observed_probability))
    else:
        standardized_residual = (observed - fitted) / np.sqrt(
            variance * np.clip(1.0 - leverage, 1e-10, None)
        )
        cook = standardized_residual**2 * leverage / (
            parameter_count * np.clip(1.0 - leverage, 1e-10, None)
        )

    leverage_reference = min(2.0 * parameter_count / n, 0.999999)
    residual_reference = 2.0
    cook_reference = 4.0 / n if cook is not None else None
    score = np.maximum(
        leverage / max(leverage_reference, np.finfo(float).eps),
        np.abs(standardized_residual) / residual_reference,
    )
    if cook is not None and cook_reference is not None:
        score = np.maximum(score, cook / cook_reference)
    candidate = score > 1.0
    return {
        "status": "available",
        "method": method,
        "leverage": leverage,
        "standardized_residual": standardized_residual,
        "cook_distance": cook,
        "priority_score": score,
        "candidate_mask": candidate,
        "thresholds": {
            "leverage": leverage_reference,
            "absolute_standardized_residual": residual_reference,
            "cook_distance": cook_reference,
            "threshold_role": "screening references, not automatic deletion rules",
        },
    }


def parameter_sensitivity(reference: Any, alternative: Any, alpha: float) -> dict[str, Any]:
    reference_params = np.asarray(reference.params, dtype=float).reshape(-1)
    alternative_params = np.asarray(alternative.params, dtype=float).reshape(-1)
    reference_se = np.asarray(reference.bse, dtype=float).reshape(-1)
    reference_p = np.asarray(reference.pvalues, dtype=float).reshape(-1)
    alternative_p = np.asarray(alternative.pvalues, dtype=float).reshape(-1)
    if not (
        reference_params.shape == alternative_params.shape == reference_se.shape
        and reference_p.shape == alternative_p.shape == reference_params.shape
    ):
        raise ValueError("Case-deletion refit changed the parameter layout")
    standardized_change = np.abs(alternative_params - reference_params) / np.clip(
        np.abs(reference_se), 1e-12, None
    )
    return {
        "max_standardized_parameter_change": float(np.max(standardized_change)),
        "sign_flip_count": int(np.sum(np.sign(reference_params) != np.sign(alternative_params))),
        "significance_flip_count": int(
            np.sum((reference_p < alpha) != (alternative_p < alpha))
        ),
    }


def finite_exp(value: float, label: str) -> float:
    try:
        transformed = math.exp(float(value))
    except OverflowError:
        fail(f"Exponentiated {label} overflowed; results were not written.")
    if not math.isfinite(transformed):
        fail(f"Exponentiated {label} is nonfinite; results were not written.")
    return transformed


def association_test(table: pd.DataFrame) -> tuple[float, float, float]:
    chi2, p_value, _, _ = stats.chi2_contingency(table.to_numpy())
    n = float(table.to_numpy().sum())
    denominator = max(min(table.shape[0] - 1, table.shape[1] - 1), 1)
    cramers_v = math.sqrt(max(float(chi2), 0.0) / max(n * denominator, 1.0))
    return float(chi2), float(p_value), float(cramers_v)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    spec_path = Path(args.spec).resolve()
    prep_path = Path(args.preparation_log).resolve()
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    targets = [
        output_dir / "model-results.csv",
        output_dir / "model-diagnostics.csv",
        output_dir / "factor-omnibus-tests.csv",
        output_dir / "continuous-shape-tests.csv",
        output_dir / "influence-diagnostics.csv",
        output_dir / "iia-check.json",
        output_dir / "proportional-odds-check.json",
        output_dir / "count-dispersion-check.json",
        output_dir / "negative-binomial-need-check.json",
        output_dir / "zero-inflation-check.json",
        output_dir / "predictive-validation.json",
        output_dir / "model-summary.json",
        output_dir / "analysis-run-log.json",
        output_dir / "analysis-summary.md",
        output_dir / "learning-prompts.json",
    ]
    if not args.overwrite and any(path.exists() for path in targets):
        fail("Analysis outputs already exist; use --overwrite only after explicit approval")
    if not data_path.is_file() or not spec_path.is_file() or not prep_path.is_file():
        fail("Data, model specification, and preparation log must all exist")
    spec = read_json(spec_path, "model specification")
    prep = read_json(prep_path, "preparation log")
    if (
        spec.get("status") != "approved"
        or spec.get("user_confirmed") is not True
        or not isinstance(spec.get("approval"), dict)
    ):
        fail("Model specification is not explicitly approved")
    if spec.get("modeling_executed") is not False:
        fail("Model specification must contain modeling_executed: false")
    collinearity_screen = spec.get("collinearity_screen", {})
    if (
        isinstance(collinearity_screen, dict)
        and collinearity_screen.get("model_fitting_allowed") is False
    ):
        fail("Approved model contains unresolved deterministic duplicate encodings")
    workflow_support = spec.get("workflow_support", {})
    if (
        not isinstance(workflow_support, dict)
        or workflow_support.get("status") != "supported"
        or workflow_support.get("execution_allowed") is not True
    ):
        fail("Approved specification does not permit this bundled workflow to execute")
    if prep.get("data_preparation_executed") is not True or prep.get("modeling_executed") is not False:
        fail("Preparation log is not eligible for modeling")
    missingness_screen = prep.get("missingness_bias_screen")
    missingness_contract = prep.get("missingness_conclusion_contract")
    if not isinstance(missingness_screen, dict) or not isinstance(missingness_contract, dict):
        fail("Preparation log lacks the approved missingness-impact contract")
    if (
        missingness_screen.get("status") == "review-required"
        and missingness_contract.get("scope") != "analyzed-sample-only"
        and missingness_contract.get("model_estimate_sensitivity_completed") is not True
    ):
        fail("Missing selected-field data require analyzed-sample-only conclusions or completed model sensitivity evidence")
    provenance = spec.get("provenance", {})
    if sha256_file(data_path) != provenance.get("cleaned_data_sha256"):
        fail("Cleaned data hash does not match the approved model specification")
    if sha256_file(prep_path) != provenance.get("preparation_log_sha256"):
        fail("Preparation log hash does not match the approved model specification")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    outcome = spec["outcome"]
    predictors = spec["predictors"]
    controls = spec["controls"]
    variables = [outcome, *predictors, *controls]
    if any(column not in df.columns for column in variables):
        fail("Cleaned data no longer contains all approved variables")
    frame = df[variables].copy()
    excluded_rows = [int(index) + 2 for index in frame.index[frame.isna().any(axis=1)]]
    frame = frame.dropna()
    model_type = spec["model_type"]
    categorical = set(spec.get("categorical_columns", []))
    categorical_predictors = categorical.intersection({*predictors, *controls})
    if categorical != categorical_predictors:
        fail("Approved categorical_columns contains a non-predictor/non-control column")
    raw_references = spec.get("categorical_reference_categories")
    if not isinstance(raw_references, dict):
        fail("Approved specification is missing categorical_reference_categories")
    categorical_references = {
        str(column): str(category) for column, category in raw_references.items()
    }
    continuous_columns = set([*predictors, *controls]) - categorical_predictors
    continuous_form_specs = validate_continuous_form_specifications(
        spec.get("continuous_functional_forms"), continuous_columns, model_type
    )
    if model_type not in {"logistic", "multinomial-logistic", "ordinal-logistic"}:
        validate_sparse_predictor_levels(frame, categorical_predictors)
    x = frame[[*predictors, *controls]].copy()
    for column in x.columns:
        if column not in categorical:
            x[column] = pd.to_numeric(x[column], errors="coerce")
            if x[column].isna().any():
                fail(f"Approved numeric predictor {column!r} is not numeric")
    x = apply_continuous_forms(x, continuous_form_specs)
    x = encode_categorical_predictors(
        x, categorical_predictors, categorical_references
    )
    x_no_constant = x.astype(float)
    x = sm.add_constant(x_no_constant, has_constant="add")
    design = x_no_constant if model_type == "ordinal-logistic" else x
    if len(frame) <= int(spec.get("estimated_parameter_count", len(design.columns))):
        fail("Model has no positive residual degrees of freedom")
    runtime_rank = int(np.linalg.matrix_rank(design.to_numpy()))
    if runtime_rank < len(design.columns):
        fail("Design matrix is rank-deficient; do not silently remove terms")
    approved_preflight = spec.get("design_matrix_preflight", {})
    if (
        not isinstance(approved_preflight, dict)
        or approved_preflight.get("status") != "passed"
        or approved_preflight.get("model_fitting_allowed") is not True
        or approved_preflight.get("row_count") != len(design)
        or approved_preflight.get("column_count") != len(design.columns)
        or approved_preflight.get("rank") != runtime_rank
        or approved_preflight.get("design_columns") != [str(column) for column in design.columns]
    ):
        fail("Runtime design matrix differs from the preflighted matrix approved by the user")
    runtime_category_support = screen_category_support(
        frame,
        outcome=outcome,
        categorical_predictors=categorical_predictors,
        model_type=model_type,
        parameters_per_outcome_equation=len(x.columns),
        design_matrix=x_no_constant,
        positive_class=(str(spec.get("positive_class")) if model_type == "logistic" else None),
    )
    if runtime_category_support != spec.get("category_support_screen"):
        fail("Runtime category-support evidence differs from the approved model specification")
    if runtime_category_support.get("model_fitting_allowed") is False:
        fail("Approved model still contains unresolved sparse-category or separation risk")
    approved_iia = spec.get("iia_check")
    if not isinstance(approved_iia, dict):
        fail("Approved specification is missing the IIA check")
    if model_type == "multinomial-logistic":
        try:
            runtime_iia = screen_iia(
                frame,
                outcome,
                x,
                [str(value) for value in spec.get("outcome_categories", [])],
                1.0 - float(spec.get("confidence_level", 0.95)),
            )
        except Exception as exc:
            fail(
                "Runtime IIA check could not be evaluated reliably; ordinary multinomial "
                f"logistic was stopped: {type(exc).__name__}: {exc}"
            )
        if not iia_materially_matches(approved_iia, runtime_iia):
            fail("Runtime IIA evidence differs from the approved specification")
        if runtime_iia.get("model_fitting_allowed") is not True:
            fail("Ordinary multinomial logistic is blocked by the IIA sensitivity check")
    else:
        runtime_iia = iia_not_applicable()
        if approved_iia != runtime_iia:
            fail("Non-multinomial model has an inconsistent IIA record")
    approved_proportional_odds = spec.get("proportional_odds_check")
    if not isinstance(approved_proportional_odds, dict):
        fail("Approved specification is missing the proportional-odds check")
    if model_type == "ordinal-logistic":
        try:
            runtime_proportional_odds = screen_proportional_odds(
                frame,
                outcome,
                x_no_constant,
                [str(value) for value in spec.get("outcome_categories", [])],
                1.0 - float(spec.get("confidence_level", 0.95)),
            )
        except Exception as exc:
            fail(
                "Runtime proportional-odds check could not be evaluated reliably; "
                f"ordinary ordinal logistic was stopped: {type(exc).__name__}: {exc}"
            )
        if not materially_matches(approved_proportional_odds, runtime_proportional_odds):
            fail("Runtime proportional-odds evidence differs from the approved specification")
        if runtime_proportional_odds.get("model_fitting_allowed") is not True:
            fail("Ordinary ordinal logistic is blocked by the proportional-odds check")
    else:
        runtime_proportional_odds = not_applicable()
        if approved_proportional_odds != runtime_proportional_odds:
            fail("Non-ordinal model has an inconsistent proportional-odds record")
    approved_count_dispersion = spec.get("count_dispersion_check")
    if not isinstance(approved_count_dispersion, dict):
        fail("Approved specification is missing the count-dispersion check")
    if model_type == "poisson":
        try:
            runtime_count_dispersion = screen_count_dispersion(
                frame,
                outcome,
                x,
                1.0 - float(spec.get("confidence_level", 0.95)),
            )
        except Exception as exc:
            fail(
                "Runtime Poisson overdispersion check could not be evaluated reliably; "
                f"ordinary Poisson was stopped: {type(exc).__name__}: {exc}"
            )
        if not count_dispersion_materially_matches(
            approved_count_dispersion, runtime_count_dispersion
        ):
            fail("Runtime count-dispersion evidence differs from the approved specification")
        if runtime_count_dispersion.get("model_fitting_allowed") is not True:
            fail("Ordinary Poisson is blocked by the adjusted overdispersion check")
    else:
        runtime_count_dispersion = count_dispersion_not_applicable()
        if approved_count_dispersion != runtime_count_dispersion:
            fail("Non-Poisson model has an inconsistent count-dispersion record")
    approved_negative_binomial_need = spec.get("negative_binomial_need_check")
    if not isinstance(approved_negative_binomial_need, dict):
        fail("Approved specification is missing the negative-binomial need check")
    if model_type == "negative-binomial":
        try:
            runtime_negative_binomial_need = screen_negative_binomial_need(
                frame,
                outcome,
                x,
                1.0 - float(spec.get("confidence_level", 0.95)),
            )
        except Exception as exc:
            fail(
                "Runtime negative-binomial need check could not be evaluated reliably; "
                f"the model was stopped: {type(exc).__name__}: {exc}"
            )
        if not negative_binomial_need_materially_matches(
            approved_negative_binomial_need, runtime_negative_binomial_need
        ):
            fail("Runtime negative-binomial need evidence differs from the approved specification")
        if runtime_negative_binomial_need.get("model_fitting_allowed") is not True:
            fail("Negative-binomial regression is blocked because extra dispersion is unsupported")
    else:
        runtime_negative_binomial_need = negative_binomial_need_not_applicable()
        if approved_negative_binomial_need != runtime_negative_binomial_need:
            fail("Non-negative-binomial model has an inconsistent need-check record")
    approved_zero_inflation = spec.get("zero_inflation_check")
    if not isinstance(approved_zero_inflation, dict):
        fail("Approved specification is missing the excess-zero check")
    if model_type in {"poisson", "negative-binomial"}:
        try:
            runtime_zero_inflation = screen_zero_inflation(
                frame,
                outcome,
                x,
                model_type,
                1.0 - float(spec.get("confidence_level", 0.95)),
            )
        except Exception as exc:
            fail(
                "Runtime excess-zero check could not be evaluated reliably; "
                f"ordinary count regression was stopped: {type(exc).__name__}: {exc}"
            )
        if not zero_inflation_materially_matches(
            approved_zero_inflation, runtime_zero_inflation
        ):
            fail("Runtime excess-zero evidence differs from the approved specification")
        if runtime_zero_inflation.get("model_fitting_allowed") is not True:
            fail("Ordinary count regression is blocked by the excess-zero check")
    else:
        runtime_zero_inflation = zero_inflation_not_applicable()
        if approved_zero_inflation != runtime_zero_inflation:
            fail("Non-count model has an inconsistent excess-zero record")

    robust = spec.get("robust_se", "HC3")
    confidence = float(spec.get("confidence_level", 0.95))
    y_raw = frame[outcome]
    y_numeric: pd.Series | None = None
    if model_type in {"ols", "poisson", "negative-binomial"}:
        y_numeric = pd.to_numeric(y_raw, errors="coerce")
        if y_numeric.isna().any():
            fail("Outcome is not numeric after approved preparation")
    y: pd.Series | None = None
    fitted_probabilities: np.ndarray | None = None
    outcome_categories = [str(value) for value in (spec.get("outcome_categories") or [])]
    observed_outcome_categories = sorted(y_raw.astype(str).unique().tolist())
    if model_type == "logistic":
        positive = str(spec.get("positive_class"))
        outcome_reference = str(spec.get("outcome_reference_class"))
        if set([positive, outcome_reference]) != set(observed_outcome_categories):
            fail(
                "Approved logistic target/reference classes do not match the observed outcome categories"
            )
    elif model_type == "multinomial-logistic":
        if not outcome_categories or str(spec.get("reference_class")) != outcome_categories[0]:
            fail("Approved multinomial reference_class must be the first outcome category")
    fit_warnings: list[python_warnings.WarningMessage] = []
    try:
        with python_warnings.catch_warnings(record=True) as captured_warnings:
            python_warnings.simplefilter("always")
            if model_type == "ols":
                assert y_numeric is not None
                base_result = sm.OLS(y_numeric.astype(float), x).fit()
                result = base_result if robust == "nonrobust" else base_result.get_robustcov_results(cov_type=robust)
            elif model_type == "logistic":
                positive = str(spec.get("positive_class"))
                y = (y_raw.astype(str) == positive).astype(int)
                validate_observed_separation(x_no_constant, y, model_type)
                result = sm.Logit(y, x).fit(disp=False, cov_type="nonrobust" if robust == "nonrobust" else robust)
            elif model_type == "poisson":
                assert y_numeric is not None
                y = y_numeric.astype(float)
                validate_count_separation(x_no_constant, y, model_type)
                result = sm.GLM(y, x, family=sm.families.Poisson()).fit(
                    cov_type="nonrobust" if robust == "nonrobust" else robust
                )
            elif model_type == "negative-binomial":
                assert y_numeric is not None
                y = y_numeric.astype(float)
                validate_count_separation(x_no_constant, y, model_type)
                result = sm.NegativeBinomial(y, x).fit(
                    disp=False,
                    cov_type="nonrobust" if robust == "nonrobust" else robust,
                )
            elif model_type == "multinomial-logistic":
                if len(outcome_categories) < 3:
                    fail("Approved multinomial model is missing outcome_categories")
                y = outcome_codes(y_raw, outcome_categories, "Multinomial outcome")
                validate_observed_separation(x_no_constant, y, model_type)
                result = sm.MNLogit(y, x).fit(
                    method="newton",
                    maxiter=200,
                    disp=False,
                    cov_type="nonrobust" if robust == "nonrobust" else robust,
                )
                fitted_probabilities = np.asarray(result.predict(x), dtype=float)
            elif model_type == "ordinal-logistic":
                if len(outcome_categories) < 3:
                    fail("Approved ordinal model is missing outcome_categories")
                y = outcome_codes(y_raw, outcome_categories, "Ordinal outcome")
                validate_observed_separation(x_no_constant, y, model_type)
                ordered_model = OrderedModel(
                    y,
                    x_no_constant,
                    distr="logit",
                )
                result = ordered_model.fit(
                    method="bfgs",
                    maxiter=300,
                    disp=False,
                    cov_type="nonrobust" if robust == "nonrobust" else robust,
                )
                fitted_probabilities = np.asarray(result.model.predict(result.params), dtype=float)
            else:
                fail(f"Unsupported approved model type: {model_type}")
            fit_warnings = list(captured_warnings)
    except Exception as exc:
        fail(f"Model fitting failed without fallback: {type(exc).__name__}: {exc}")

    critical_warning_types = (
        ConvergenceWarning,
        HessianInversionWarning,
        PerfectSeparationWarning,
        SingularMatrixWarning,
    )
    critical_warnings = [
        warning
        for warning in fit_warnings
        if issubclass(warning.category, critical_warning_types)
    ]
    if critical_warnings:
        messages = "; ".join(str(warning.message) for warning in critical_warnings)
        fail(f"Model fitting emitted a critical warning and was stopped: {messages}")
    validate_fitted_model(result, model_type, fitted_probabilities)

    results_rows: list[dict[str, Any]] = []
    if model_type == "multinomial-logistic":
        params = np.asarray(result.params, dtype=float)
        bse = np.asarray(result.bse, dtype=float)
        pvalues = np.asarray(result.pvalues, dtype=float)
        ci_values = np.asarray(result.conf_int(alpha=1 - confidence), dtype=float)
        for category_index, category in enumerate(outcome_categories[1:]):
            for term_index, term in enumerate(x.columns):
                confidence_row = ci_values[
                    category_index * len(x.columns) + term_index
                ]
                estimate = float(params[term_index, category_index])
                low = float(confidence_row[0])
                high = float(confidence_row[1])
                results_rows.append(
                    {
                        "outcome_category": category,
                        "reference_class": outcome_categories[0],
                        "term": term,
                        "term_type": "coefficient",
                        "estimate": estimate,
                        "std_error": float(bse[term_index, category_index]),
                        "statistic": estimate / float(bse[term_index, category_index]),
                        "p_value": float(pvalues[term_index, category_index]),
                        "ci_low": low,
                        "ci_high": high,
                        "exp_estimate": finite_exp(estimate, "coefficient"),
                        "exp_ci_low": finite_exp(low, "confidence bound"),
                        "exp_ci_high": finite_exp(high, "confidence bound"),
                    }
                )
    else:
        params = np.asarray(result.params, dtype=float)
        bse = np.asarray(result.bse, dtype=float)
        pvalues = np.asarray(result.pvalues, dtype=float)
        statistics = params / bse
        ci = np.asarray(result.conf_int(alpha=1 - confidence), dtype=float)
        term_names = (
            list(map(str, result.params.index))
            if isinstance(result.params, pd.Series)
            else list(map(str, getattr(result.model, "exog_names", design.columns)))
        )
        for index, term in enumerate(term_names):
            term_type = "coefficient"
            if model_type == "ordinal-logistic" and "/" in term:
                term_type = "threshold"
            if model_type == "negative-binomial" and term == "alpha":
                term_type = "dispersion"
            row: dict[str, Any] = {
                "term": term,
                "term_type": term_type,
                "estimate": params[index],
                "std_error": bse[index],
                "statistic": statistics[index],
                "p_value": pvalues[index],
                "ci_low": ci[index, 0],
                "ci_high": ci[index, 1],
            }
            if model_type in {
                "logistic",
                "poisson",
                "negative-binomial",
                "ordinal-logistic",
            } and term_type == "coefficient":
                row.update(
                    {
                        "exp_estimate": finite_exp(params[index], "coefficient"),
                        "exp_ci_low": finite_exp(ci[index, 0], "confidence bound"),
                        "exp_ci_high": finite_exp(ci[index, 1], "confidence bound"),
                    }
                )
            results_rows.append(row)
    coefficient_row_indices = [
        index
        for index, row in enumerate(results_rows)
        if row.get("term_type") == "coefficient" and row.get("term") != "const"
    ]
    adjusted_coefficient_p = benjamini_hochberg(
        [float(results_rows[index]["p_value"]) for index in coefficient_row_indices]
    )
    for index, adjusted_p in zip(
        coefficient_row_indices, adjusted_coefficient_p, strict=True
    ):
        results_rows[index]["p_value_adjusted_bh"] = adjusted_p
        results_rows[index]["multiplicity_family"] = "all-non-intercept-coefficients"

    factor_design_columns = list(
        map(str, x_no_constant.columns if model_type == "ordinal-logistic" else x.columns)
    )
    factor_tests = categorical_omnibus_tests(
        result,
        model_type,
        factor_design_columns,
        categorical_references,
    )
    factor_test_map = {str(item["factor"]): item for item in factor_tests}
    continuous_shape_tests: list[dict[str, Any]] = []
    fitted_design_columns = list(map(str, x.columns))
    for column, form_spec in sorted(continuous_form_specs.items()):
        terms = [str(term) for term in form_spec["term_names"]]
        overall_statistic, overall_df, overall_p = joint_wald_test(
            result, fitted_design_columns, terms, f"continuous predictor {column}"
        )
        nonlinear_terms = terms[1:]
        item: dict[str, Any] = {
            "variable": column,
            "form": form_spec["form"],
            "overall_test": "joint-wald-chi-square",
            "overall_statistic": overall_statistic,
            "overall_degrees_of_freedom": overall_df,
            "overall_p_value": overall_p,
            "nonlinear_terms": nonlinear_terms,
            "nonlinear_test": "joint-wald-chi-square" if nonlinear_terms else "not-applicable",
            "nonlinear_statistic": None,
            "nonlinear_degrees_of_freedom": 0,
            "nonlinear_p_value": None,
        }
        if nonlinear_terms:
            statistic, degrees_of_freedom, p_value = joint_wald_test(
                result, fitted_design_columns, nonlinear_terms, f"nonlinear component of {column}"
            )
            item.update(
                {
                    "nonlinear_statistic": statistic,
                    "nonlinear_degrees_of_freedom": degrees_of_freedom,
                    "nonlinear_p_value": p_value,
                }
            )
        continuous_shape_tests.append(item)
    nonlinear_test_items = [
        item for item in continuous_shape_tests if item["nonlinear_p_value"] is not None
    ]
    overall_adjusted = benjamini_hochberg(
        [float(item["overall_p_value"]) for item in continuous_shape_tests]
    )
    for item, adjusted_p in zip(continuous_shape_tests, overall_adjusted, strict=True):
        item["overall_p_value_adjusted_bh"] = adjusted_p
    nonlinear_adjusted = benjamini_hochberg(
        [float(item["nonlinear_p_value"]) for item in nonlinear_test_items]
    )
    for item, adjusted_p in zip(nonlinear_test_items, nonlinear_adjusted, strict=True):
        item["nonlinear_p_value_adjusted_bh"] = adjusted_p
    alpha = 1.0 - confidence
    for row in results_rows:
        if row.get("term_type") != "coefficient" or row.get("term") == "const":
            continue
        factor = categorical_factor_for_term(
            row.get("term"), categorical_references
        )
        adjusted_p = float(row["p_value_adjusted_bh"])
        supported = adjusted_p < alpha
        row["categorical_factor"] = factor
        if factor is not None:
            factor_test = factor_test_map[factor]
            row["factor_omnibus_p_value"] = factor_test["p_value"]
            row["factor_omnibus_p_value_adjusted_bh"] = factor_test[
                "p_value_adjusted_bh"
            ]
            supported = supported and float(
                factor_test["p_value_adjusted_bh"]
            ) < alpha
        row["multiplicity_supported"] = supported

    results_df = pd.DataFrame(results_rows)
    term_to_continuous_source = {
        str(term): column
        for column, form_spec in continuous_form_specs.items()
        for term in form_spec["term_names"]
    }
    results_df["continuous_source_variable"] = results_df["term"].map(
        term_to_continuous_source
    )
    results_df["continuous_functional_form"] = results_df[
        "continuous_source_variable"
    ].map(
        lambda column: continuous_form_specs.get(str(column), {}).get("form")
        if pd.notna(column)
        else None
    )
    results_df["nonlinear_basis_term"] = results_df["term"].map(
        lambda term: bool(
            term in term_to_continuous_source
            and str(term) != term_to_continuous_source[str(term)]
        )
    )
    shape_test_map = {str(item["variable"]): item for item in continuous_shape_tests}
    results_df["continuous_overall_p_value_adjusted_bh"] = results_df[
        "continuous_source_variable"
    ].map(
        lambda column: shape_test_map.get(str(column), {}).get("overall_p_value_adjusted_bh")
        if pd.notna(column)
        else None
    )
    results_df["continuous_nonlinear_p_value_adjusted_bh"] = results_df[
        "continuous_source_variable"
    ].map(
        lambda column: shape_test_map.get(str(column), {}).get("nonlinear_p_value_adjusted_bh")
        if pd.notna(column)
        else None
    )
    reference_columns = sorted(categorical_references, key=len, reverse=True)

    def predictor_reference_for_term(term: Any) -> str | None:
        term_text = str(term)
        for column in reference_columns:
            if term_text.startswith(f"{column}_"):
                return categorical_references[column]
        return None

    results_df["predictor_reference_category"] = results_df["term"].map(
        predictor_reference_for_term
    )

    if model_type == "multinomial-logistic":
        assert y is not None and fitted_probabilities is not None
        predicted_class = np.argmax(fitted_probabilities, axis=1)
        fitted = np.max(fitted_probabilities, axis=1)
        residual_fallback = (
            (predicted_class == np.asarray(y, dtype=int)).astype(float) - fitted
        )
    elif fitted_probabilities is not None:
        category_scores = np.arange(fitted_probabilities.shape[1], dtype=float)
        fitted = fitted_probabilities @ category_scores
    else:
        fitted = np.asarray(result.predict(x), dtype=float)
    if model_type == "multinomial-logistic":
        pass
    elif model_type == "logistic":
        assert y is not None
        residual_fallback = np.asarray(y, dtype=float) - fitted
    elif model_type == "ordinal-logistic":
        assert y is not None
        residual_fallback = np.asarray(y, dtype=float) - fitted
    else:
        assert y_numeric is not None
        residual_fallback = np.asarray(y_numeric, dtype=float) - fitted
    try:
        residual_candidate = result.resid
    except Exception:
        try:
            residual_candidate = result.resid_deviance
        except Exception:
            residual_candidate = residual_fallback
    resid = np.asarray(residual_candidate, dtype=float)
    diagnostics: list[dict[str, Any]] = []
    diagnostics.append(
        {
            "category": "count-assumption",
            "metric": "poisson_overdispersion_status",
            "value": runtime_count_dispersion.get("status"),
        }
    )
    if model_type == "poisson":
        diagnostics.append(
            {
                "category": "count-assumption",
                "metric": "poisson_overdispersion_one_sided_p_value",
                "value": scalar(runtime_count_dispersion.get("p_value")),
            }
        )
    bic_value = getattr(result, "bic_llf", None) if model_type == "poisson" else getattr(result, "bic", None)
    metrics: dict[str, Any] = {
        "n": len(frame),
        "parameters": len(results_rows),
        "df_resid": scalar(result.df_resid),
        "aic": scalar(getattr(result, "aic", None)),
        "bic": scalar(bic_value),
        "llr_p_value": scalar(getattr(result, "llr_pvalue", None)),
    }
    if model_type == "ols":
        metrics.update(
            {
                "r_squared": scalar(getattr(result, "rsquared", getattr(base_result, "rsquared", None))),
                "adjusted_r_squared": scalar(getattr(result, "rsquared_adj", getattr(base_result, "rsquared_adj", None))),
                "rmse": float(np.sqrt(np.mean(np.square(resid)))),
                "durbin_watson": float(durbin_watson(resid)),
                "f_p_value": scalar(getattr(result, "f_pvalue", None)),
            }
        )

        try:
            bp = het_breuschpagan(resid, x.to_numpy())
            metrics["breusch_pagan_p_value"] = float(bp[1])
        except Exception:
            metrics["breusch_pagan_p_value"] = None
        try:
            metrics["jarque_bera_p_value"] = float(stats.jarque_bera(resid).pvalue)
        except Exception:
            metrics["jarque_bera_p_value"] = None
    elif model_type == "logistic":
        assert y is not None
        y_array = np.asarray(y, dtype=int)
        predicted_binary = (fitted >= 0.5).astype(int)
        calibration = calibration_table(y_array, fitted)
        metrics.update(
            {
                "log_likelihood": scalar(result.llf),
                "mcfadden_pseudo_r_squared": scalar(getattr(result, "prsquared", None)),
                "accuracy_at_0_5": float(np.mean(predicted_binary == y_array)),
                "roc_auc": auc_rank(y_array, fitted),
                "brier_score": float(np.mean(np.square(y_array - fitted))),
                "calibration_mean_absolute_error": (
                    float(np.mean(np.abs(calibration["observed"] - calibration["predicted"])))
                    if len(calibration)
                    else None
                ),
                "sensitivity_at_0_5": float(
                    np.sum((predicted_binary == 1) & (y_array == 1)) / max(np.sum(y_array == 1), 1)
                ),
                "specificity_at_0_5": float(
                    np.sum((predicted_binary == 0) & (y_array == 0)) / max(np.sum(y_array == 0), 1)
                ),
            }
        )
    elif model_type == "poisson":
        dispersion = float(result.pearson_chi2 / result.df_resid) if result.df_resid else None
        metrics.update(
            {
                "deviance": scalar(result.deviance),
                "pearson_chi2": scalar(result.pearson_chi2),
                "dispersion": dispersion,
                "count_rmse": float(np.sqrt(np.mean(np.square(np.asarray(y, dtype=float) - fitted)))),
            }
        )
    elif model_type == "negative-binomial":
        alpha_row = next(
            (row for row in results_rows if row.get("term_type") == "dispersion"),
            None,
        )
        metrics.update(
            {
                "log_likelihood": scalar(getattr(result, "llf", None)),
                "mcfadden_pseudo_r_squared": scalar(getattr(result, "prsquared", None)),
                "negative_binomial_alpha": scalar(
                    alpha_row.get("estimate") if alpha_row else None
                ),
                "count_rmse": float(np.sqrt(np.mean(np.square(np.asarray(y, dtype=float) - fitted)))),
            }
        )
    elif model_type in {"multinomial-logistic", "ordinal-logistic"}:
        assert y is not None and fitted_probabilities is not None
        predicted_class = np.argmax(fitted_probabilities, axis=1)
        probability_floor = np.clip(
            fitted_probabilities[np.arange(len(y)), np.asarray(y, dtype=int)],
            1e-12,
            1,
        )
        metrics.update(
            {
                "log_likelihood": scalar(getattr(result, "llf", None)),
                "mcfadden_pseudo_r_squared": scalar(getattr(result, "prsquared", None)),
                "classification_accuracy": float(
                    np.mean(predicted_class == np.asarray(y, dtype=int))
                ),
                "outcome_categories": len(outcome_categories),
                "multiclass_log_loss": float(-np.mean(np.log(probability_floor))),
                "ordinal_mean_absolute_category_error": (
                    float(np.mean(np.abs(predicted_class - np.asarray(y, dtype=int))))
                    if model_type == "ordinal-logistic"
                    else None
                ),
            }
        )

    validation_observed = (
        y_numeric.astype(float)
        if model_type == "ols" and y_numeric is not None
        else y
    )
    if validation_observed is None:
        fail("Predictive validation is missing the fitted outcome representation")
    predictive_validation = predictive_cross_validation(
        model_type,
        design.reset_index(drop=True),
        pd.Series(np.asarray(validation_observed)).reset_index(drop=True),
        outcome_categories,
        str(spec.get("goal", "association")),
    )
    cross_validated_scalar_predictions = predictive_validation.pop(
        "_out_of_fold_scalar_predictions", None
    )
    cross_validated_probability_predictions = predictive_validation.pop(
        "_out_of_fold_probability_predictions", None
    )
    if str(spec.get("goal")) == "prediction" and predictive_validation.get("status") != "completed":
        warnings_for_prediction = (
            "Cross-validated prediction could not be completed; do not make new-data performance claims."
        )
    else:
        warnings_for_prediction = None

    applicability = {
        "ols": {
            "normal_qq": "applicable",
            "heteroskedasticity": "applicable",
            "roc_calibration": "not-applicable",
            "count_dispersion": "not-applicable",
        },
        "logistic": {
            "normal_qq": "not-applicable",
            "heteroskedasticity": "not-applicable",
            "roc_calibration": "applicable",
            "count_dispersion": "not-applicable",
        },
        "poisson": {
            "normal_qq": "not-applicable",
            "heteroskedasticity": "not-applicable",
            "roc_calibration": "not-applicable",
            "count_dispersion": "applicable",
        },
        "negative-binomial": {
            "normal_qq": "not-applicable",
            "heteroskedasticity": "not-applicable",
            "roc_calibration": "not-applicable",
            "count_dispersion": "applicable",
        },
        "multinomial-logistic": {
            "normal_qq": "not-applicable",
            "heteroskedasticity": "not-applicable",
            "roc_calibration": "multiclass-specific",
            "count_dispersion": "not-applicable",
            "iia": "applicable",
        },
        "ordinal-logistic": {
            "normal_qq": "not-applicable",
            "heteroskedasticity": "not-applicable",
            "roc_calibration": "ordinal-specific",
            "count_dispersion": "not-applicable",
            "iia": "not-applicable",
        },
    }[model_type]

    for key, value in metrics.items():
        diagnostics.append({"category": "model", "metric": key, "value": scalar(value)})
    for key, value in applicability.items():
        diagnostics.append({"category": "applicability", "metric": key, "value": value})
    diagnostics.append(
        {
            "category": "predictive-validation",
            "metric": "status",
            "value": predictive_validation.get("status"),
        }
    )
    for key, value in predictive_validation.get("metrics", {}).items():
        diagnostics.append(
            {"category": "predictive-validation", "metric": key, "value": scalar(value)}
        )
    diagnostics.append(
        {
            "category": "multinomial-assumption",
            "metric": "iia_status",
            "value": runtime_iia.get("status"),
        }
    )
    if model_type == "multinomial-logistic":
        diagnostics.append(
            {
                "category": "multinomial-assumption",
                "metric": "iia_minimum_holm_p_value",
                "value": scalar(runtime_iia.get("minimum_adjusted_p_value")),
            }
        )
    diagnostics.append(
        {
            "category": "ordinal-assumption",
            "metric": "proportional_odds_status",
            "value": runtime_proportional_odds.get("status"),
        }
    )
    if model_type == "ordinal-logistic":
        diagnostics.append(
            {
                "category": "ordinal-assumption",
                "metric": "proportional_odds_equal_slopes_wald_p_value",
                "value": scalar(runtime_proportional_odds.get("p_value")),
            }
        )
    vif_by_term: dict[str, float] = {}
    if len(x.columns) > 2:
        for index, term in enumerate(x.columns):
            if term == "const":
                continue
            try:
                vif = float(variance_inflation_factor(x.to_numpy(), index))
            except Exception:
                vif = float("nan")
            diagnostics.append({"category": "multicollinearity", "metric": f"VIF:{term}", "value": scalar(vif)})
            if np.isfinite(vif):
                vif_by_term[str(term)] = vif

    jointly_interpreted_shape_terms = {
        str(term)
        for form_spec in continuous_form_specs.values()
        if form_spec.get("form") != "linear"
        for term in form_spec.get("term_names", [])
    }
    structural_basis_high_vif_terms = sorted(
        term
        for term, value in vif_by_term.items()
        if value >= 10 and term in jointly_interpreted_shape_terms
    )
    severe_vif_terms = sorted(
        term
        for term, value in vif_by_term.items()
        if value >= 10 and term not in jointly_interpreted_shape_terms
    )
    moderate_vif_terms = sorted(
        term
        for term, value in vif_by_term.items()
        if 5 <= value < 10 and term not in jointly_interpreted_shape_terms
    )
    severe_vif_factors = sorted(
        {
            factor
            for term in severe_vif_terms
            if (
                factor := categorical_factor_for_term(term, categorical_references)
            )
            is not None
        }
    )

    def collinearity_restricted_for_term(term: Any) -> bool:
        term_text = str(term)
        factor = categorical_factor_for_term(term_text, categorical_references)
        return term_text in severe_vif_terms or (
            factor is not None and factor in severe_vif_factors
        )

    results_df["vif"] = results_df["term"].map(vif_by_term)
    results_df["collinearity_restricted"] = results_df["term"].map(
        collinearity_restricted_for_term
    )
    results_df["shape_basis_restricted"] = results_df["nonlinear_basis_term"].fillna(False).astype(bool)
    results_df["interpretation_supported"] = results_df.apply(
        lambda row: row.get("term_type") == "coefficient"
        and row.get("term") != "const"
        and bool(row.get("multiplicity_supported"))
        and not bool(row.get("collinearity_restricted"))
        and not bool(row.get("shape_basis_restricted")),
        axis=1,
    )
    collinearity_summary = {
        "status": "severe" if severe_vif_terms else ("review" if moderate_vif_terms else "clear"),
        "plain_explanation": (
            "共线性是多个自变量包含大量重复信息；VIF越高，越难稳定拆分每个变量的独立作用。"
        ),
        "maximum_vif": max(vif_by_term.values()) if vif_by_term else None,
        "moderate_vif_threshold": 5,
        "severe_vif_threshold": 10,
        "moderate_terms": moderate_vif_terms,
        "severe_terms": severe_vif_terms,
        "joint_shape_terms_excluded_from_individual_vif_gate": structural_basis_high_vif_terms,
        "severe_factors": severe_vif_factors,
        "individual_effect_interpretation_allowed": not severe_vif_terms,
        "required_action": (
            "revise the approved variable set or retain the fit with affected individual effects suppressed"
            if severe_vif_terms
            else "none"
        ),
    }

    def refit_without_position(position: int) -> Any:
        keep = np.arange(len(frame)) != position
        x_alt = x.iloc[keep]
        x_no_constant_alt = x_no_constant.iloc[keep]
        with python_warnings.catch_warnings(record=True) as captured:
            python_warnings.simplefilter("always")
            if model_type == "ols":
                assert y_numeric is not None
                base_alt = sm.OLS(y_numeric.iloc[keep].astype(float), x_alt).fit()
                alternative = (
                    base_alt
                    if robust == "nonrobust"
                    else base_alt.get_robustcov_results(cov_type=robust)
                )
                alternative_probabilities = None
            elif model_type == "logistic":
                assert y is not None
                alternative = sm.Logit(y.iloc[keep], x_alt).fit(
                    disp=False, cov_type="nonrobust" if robust == "nonrobust" else robust
                )
                alternative_probabilities = np.asarray(alternative.predict(x_alt), dtype=float)
            elif model_type == "poisson":
                assert y is not None
                alternative = sm.GLM(
                    y.iloc[keep], x_alt, family=sm.families.Poisson()
                ).fit(cov_type="nonrobust" if robust == "nonrobust" else robust)
                alternative_probabilities = None
            elif model_type == "negative-binomial":
                assert y is not None
                alternative = sm.NegativeBinomial(y.iloc[keep], x_alt).fit(
                    disp=False, cov_type="nonrobust" if robust == "nonrobust" else robust
                )
                alternative_probabilities = None
            elif model_type == "multinomial-logistic":
                assert y is not None
                alternative = sm.MNLogit(y.iloc[keep], x_alt).fit(
                    method="newton", maxiter=200, disp=False,
                    cov_type="nonrobust" if robust == "nonrobust" else robust,
                )
                alternative_probabilities = np.asarray(alternative.predict(x_alt), dtype=float)
            elif model_type == "ordinal-logistic":
                assert y is not None
                alternative_model = OrderedModel(y.iloc[keep], x_no_constant_alt, distr="logit")
                alternative = alternative_model.fit(
                    method="bfgs", maxiter=300, disp=False,
                    cov_type="nonrobust" if robust == "nonrobust" else robust,
                )
                alternative_probabilities = np.asarray(
                    alternative.model.predict(alternative.params), dtype=float
                )
            else:
                raise ValueError(f"Unsupported case-deletion refit model: {model_type}")
        critical = [
            warning
            for warning in captured
            if issubclass(warning.category, critical_warning_types)
        ]
        if critical:
            raise ValueError("; ".join(str(warning.message) for warning in critical))
        validate_fitted_model(alternative, model_type, alternative_probabilities)
        return alternative

    influence_rows: list[dict[str, Any]] = []
    influential_rows: list[int] = []
    influence_error: str | None = None
    try:
        observed_for_influence = (
            np.asarray(y, dtype=float)
            if y is not None
            else np.asarray(y_numeric, dtype=float)
        )
        influence = influence_screen(
            model_type,
            result,
            design,
            observed_for_influence,
            fitted,
            observed_class_codes=(np.asarray(y, dtype=int) if model_type in {"multinomial-logistic", "ordinal-logistic"} else None),
            fitted_probabilities=(fitted_probabilities if model_type in {"multinomial-logistic", "ordinal-logistic"} else None),
        )
    except Exception as exc:
        influence = {
            "status": "not-evaluated",
            "method": "model-specific influence screen unavailable",
            "thresholds": {},
        }
        influence_error = f"{type(exc).__name__}: {exc}"

    if influence["status"] == "available":
        candidate_mask = np.asarray(influence["candidate_mask"], dtype=bool)
        priority_score = np.asarray(influence["priority_score"], dtype=float)
        leverage_values = np.asarray(influence["leverage"], dtype=float)
        residual_values = np.asarray(influence["standardized_residual"], dtype=float)
        cook_values = influence.get("cook_distance")
        cook_array = np.asarray(cook_values, dtype=float) if cook_values is not None else None
        candidate_positions = np.flatnonzero(candidate_mask).tolist()
        influential_rows = [int(frame.index[position]) + 2 for position in candidate_positions]
        evaluated_positions = sorted(
            candidate_positions, key=lambda position: priority_score[position], reverse=True
        )[:5]
        evaluated_set = set(evaluated_positions)
        sensitivity_by_position: dict[int, dict[str, Any]] = {}
        for position in evaluated_positions:
            try:
                alternative = refit_without_position(position)
                sensitivity_by_position[position] = {
                    "refit_status": "completed",
                    **parameter_sensitivity(result, alternative, alpha),
                    "refit_error": "",
                }
            except Exception as exc:
                sensitivity_by_position[position] = {
                    "refit_status": "failed",
                    "max_standardized_parameter_change": None,
                    "sign_flip_count": None,
                    "significance_flip_count": None,
                    "refit_error": f"{type(exc).__name__}: {exc}",
                }
        for position in candidate_positions:
            reasons: list[str] = []
            thresholds = influence["thresholds"]
            if leverage_values[position] > float(thresholds["leverage"]):
                reasons.append("high-leverage")
            if abs(residual_values[position]) > float(thresholds["absolute_standardized_residual"]):
                reasons.append("large-model-residual")
            if cook_array is not None and cook_array[position] > float(thresholds["cook_distance"]):
                reasons.append("cook-distance")
            sensitivity = sensitivity_by_position.get(position, {})
            influence_rows.append(
                {
                    "cleaned_data_row": int(frame.index[position]) + 2,
                    "candidate_reasons": ";".join(reasons),
                    "leverage": float(leverage_values[position]),
                    "standardized_residual": float(residual_values[position]),
                    "cook_distance": float(cook_array[position]) if cook_array is not None else None,
                    "priority_score": float(priority_score[position]),
                    "case_deletion_evaluated": position in evaluated_set,
                    "refit_status": sensitivity.get("refit_status", "not-selected-top-five"),
                    "max_standardized_parameter_change": sensitivity.get("max_standardized_parameter_change"),
                    "sign_flip_count": sensitivity.get("sign_flip_count"),
                    "significance_flip_count": sensitivity.get("significance_flip_count"),
                    "refit_error": sensitivity.get("refit_error", ""),
                }
            )
        completed_refits = sum(row["refit_status"] == "completed" for row in influence_rows)
        failed_refits = sum(row["refit_status"] == "failed" for row in influence_rows)
        sensitivity_status = (
            "not-required-no-candidates"
            if not candidate_positions
            else ("completed" if failed_refits == 0 else ("partial" if completed_refits else "not-evaluated"))
        )
        maximum_change = max(
            (
                float(row["max_standardized_parameter_change"])
                for row in influence_rows
                if row.get("max_standardized_parameter_change") is not None
            ),
            default=None,
        )
        sign_flip_total = sum(int(row.get("sign_flip_count") or 0) for row in influence_rows)
        significance_flip_total = sum(
            int(row.get("significance_flip_count") or 0) for row in influence_rows
        )
        influence_summary = {
            "status": "available",
            "method": influence["method"],
            "thresholds": influence["thresholds"],
            "candidate_count": len(candidate_positions),
            "candidate_cleaned_data_rows": influential_rows,
            "case_deletion_policy": "refit once per candidate for at most the five highest priority candidates; original fit is retained",
            "case_deletion_evaluated_count": len(evaluated_positions),
            "case_deletion_completed_count": completed_refits,
            "case_deletion_failed_count": failed_refits,
            "sensitivity_status": sensitivity_status,
            "maximum_standardized_parameter_change": maximum_change,
            "sign_flip_count": sign_flip_total,
            "significance_flip_count": significance_flip_total,
            "automatic_deletion_performed": False,
        }
        diagnostics.extend(
            [
                {"category": "influence", "metric": "assessment_status", "value": "available"},
                {"category": "influence", "metric": "candidate_count", "value": len(candidate_positions)},
                {"category": "influence", "metric": "case_deletion_evaluated_count", "value": len(evaluated_positions)},
                {"category": "influence", "metric": "max_standardized_parameter_change", "value": scalar(maximum_change)},
            ]
        )
    else:
        influence_summary = {
            "status": "not-evaluated",
            "method": influence["method"],
            "reason": influence_error,
            "candidate_count": None,
            "candidate_cleaned_data_rows": [],
            "case_deletion_evaluated_count": 0,
            "sensitivity_status": "not-evaluated",
            "automatic_deletion_performed": False,
        }
        diagnostics.append(
            {"category": "influence", "metric": "assessment_status", "value": "not-evaluated"}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_academic_style()

    primary_predictor = str(predictors[0])
    predictor_numeric = pd.to_numeric(frame[primary_predictor], errors="coerce")
    relationship_prompt: dict[str, Any]
    if model_type == "multinomial-logistic":
        outcome_labels = y_raw.astype(str)
        if predictor_numeric.notna().all():
            unique_count = int(predictor_numeric.nunique())
            if unique_count < 2:
                fail(f"Primary predictor {primary_predictor!r} has no usable variation")
            quantiles = min(5, unique_count)
            preview_groups = pd.qcut(
                predictor_numeric,
                q=quantiles,
                duplicates="drop",
            ).astype(str)
            preview_group_label = f"{primary_predictor}分位组"
        else:
            preview_groups = frame[primary_predictor].astype(str)
            preview_group_label = primary_predictor

        association_table = pd.crosstab(preview_groups, outcome_labels)
        association_table = association_table.reindex(
            columns=outcome_categories,
            fill_value=0,
        )
        if association_table.shape[0] < 2 or association_table.shape[1] < 2:
            fail("Multinomial relationship preview requires at least two predictor groups and outcome categories")
        chi2_value, association_p, cramers_v = association_test(association_table)
        plot_table = association_table
        if len(plot_table) > 12:
            keep = plot_table.sum(axis=1).nlargest(12).index
            plot_table = plot_table.loc[keep]
        proportions = plot_table.div(plot_table.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(10.2, 6.2))
        proportions.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=plt.cm.Blues(np.linspace(0.45, 0.9, len(proportions.columns))),
            width=0.78,
        )
        ax.set_xlabel(preview_group_label)
        ax.set_ylabel("结果类别比例")
        ax.set_ylim(0, 1)
        ax.set_title("先观察：不同组中的结果类别构成", fontweight="bold", pad=14)
        ax.legend(title=outcome, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=30)
        polish_axes(ax, grid_axis="y")
        save_figure(fig, figures_dir / "relationship-preview.png")
        evidence_answer = (
            "类别比例存在明显差异"
            if association_p < 0.05 and cramers_v >= 0.1
            else "类别比例大致相近"
        )
        relationship_prompt = {
            "figure": "figures/relationship-preview.png",
            "question": (
                f"先不看模型结果：不同{preview_group_label}中的{outcome}类别比例是否存在明显差异？"
            ),
            "options": ["类别比例存在明显差异", "类别比例大致相近", "仅凭图无法判断"],
            "evidence_answer": evidence_answer,
            "evidence_feedback": (
                f"整体列联检验 χ²={chi2_value:.3f}，p={association_p:.4g}，"
                f"Cramér's V={cramers_v:.3f}，因此更接近“{evidence_answer}”。"
                + (
                    "无序类别没有高低方向，这里只比较各组的类别构成，不解释为正相关或负相关。"
                    if model_type == "multinomial-logistic"
                    else "有序类别按已批准顺序展示，但图形不假定相邻等级间距相等。"
                )
            ),
        }
    elif model_type == "logistic":
        assert y is not None
        positive_label = str(spec.get("positive_class"))
        if predictor_numeric.notna().all():
            unique_count = int(predictor_numeric.nunique())
            groups = pd.qcut(
                predictor_numeric,
                q=min(8, unique_count),
                duplicates="drop",
            )
            preview = pd.DataFrame({"group": groups, "positive": np.asarray(y, dtype=float)})
            rates = preview.groupby("group", observed=False)["positive"].mean()
            fig, ax = plt.subplots(figsize=(9.2, 5.8))
            positions = np.arange(len(rates))
            ax.plot(positions, rates.to_numpy(), marker="o", linewidth=2.2, color=PRIMARY_BLUE_GRAY)
            ax.set_xticks(positions, [str(value) for value in rates.index], rotation=25, ha="right")
            ax.set_xlabel(f"{primary_predictor}分组")
        else:
            preview = pd.DataFrame(
                {"group": frame[primary_predictor].astype(str), "positive": np.asarray(y, dtype=float)}
            )
            rates = preview.groupby("group")["positive"].agg(["mean", "size"]).sort_values("size", ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(9.2, 5.8))
            rates["mean"].sort_values().plot.barh(ax=ax, color=PRIMARY_BLUE_GRAY)
            ax.set_xlabel("正类比例")
            ax.set_ylabel(primary_predictor)
        ticks = np.linspace(0, 1, 6)
        if predictor_numeric.notna().all():
            ax.set_ylim(0, 1)
            ax.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
            ax.set_ylabel(f"{outcome}={positive_label} 的比例")
            polish_axes(ax, grid_axis="y")
        else:
            ax.set_xlim(0, 1)
            ax.set_xticks(ticks, [f"{value:.0%}" for value in ticks])
            polish_axes(ax, grid_axis="x")
        ax.set_title("先观察：不同因素水平的正类比例", fontweight="bold", pad=14)
        save_figure(fig, figures_dir / "relationship-preview.png")
        relationship_prompt = {
            "figure": "figures/relationship-preview.png",
            "question": f"不同{primary_predictor}水平下，{outcome}={positive_label}的比例是否明显不同？",
            "options": ["比例明显不同", "比例大致相近", "仅凭图无法判断"],
            "evidence_answer": "仅凭图无法判断",
            "evidence_feedback": "图中展示的是未经其他因素调整的正类比例；正式判断还需结合 Logistic 系数、置信区间和模型校准。",
        }
    elif predictor_numeric.notna().all():
        if y_numeric is not None:
            preview_y = y_numeric.astype(float)
            preview_ylabel = outcome
        else:
            assert y is not None
            preview_y = y.astype(float)
            preview_ylabel = f"{outcome}（类别顺序编码，仅供观察）"
        rho, rho_p = stats.spearmanr(predictor_numeric, preview_y)
        direction = (
            "正相关"
            if np.isfinite(rho) and rho >= 0.1
            else ("负相关" if np.isfinite(rho) and rho <= -0.1 else "没有明显单调关系")
        )
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        ax.scatter(
            predictor_numeric,
            preview_y,
            s=48,
            alpha=0.7,
            color=PRIMARY_BLUE_GRAY,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel(primary_predictor)
        ax.set_ylabel(preview_ylabel)
        ax.set_title("先观察：主要因素与结果的原始关系", fontweight="bold", pad=14)
        polish_axes(ax, grid_axis="both")
        save_figure(fig, figures_dir / "relationship-preview.png")
        relationship_prompt = {
            "figure": "figures/relationship-preview.png",
            "question": "先不看模型结果：从图中看，你认为二者大致是正相关、负相关，还是没有明显关系？",
            "options": ["正相关", "负相关", "没有明显关系"],
            "evidence_answer": direction,
            "evidence_feedback": (
                f"原始数据的 Spearman 相关系数为 {float(rho):.3f}"
                f"（p={float(rho_p):.4g}），因此更接近“{direction}”。"
                "这是未控制其他因素的观察，不能替代后续模型和诊断。"
            ),
        }
    else:
        group_counts = frame.groupby(primary_predictor, dropna=False).size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        group_counts.head(12).sort_values().plot.barh(ax=ax, color=PRIMARY_BLUE_GRAY)
        ax.set_xlabel("记录数")
        ax.set_ylabel(primary_predictor)
        ax.set_title("先观察：主要因素的类别分布", fontweight="bold", pad=14)
        polish_axes(ax, grid_axis="x")
        save_figure(fig, figures_dir / "relationship-preview.png")
        relationship_prompt = {
            "figure": "figures/relationship-preview.png",
            "question": "先不看模型结果：这些类别的样本量是否大致均衡？",
            "options": ["大致均衡", "明显不均衡", "仅凭图无法判断"],
            "evidence_answer": (
                "大致均衡"
                if len(group_counts) > 1 and group_counts.max() / max(group_counts.min(), 1) < 2
                else "明显不均衡"
            ),
            "evidence_feedback": (
                f"最大类别有 {int(group_counts.max())} 条，最小类别有 {int(group_counts.min())} 条。"
                "类别不均衡可能让小类别的估计更不稳定，需要结合置信区间解释。"
            ),
        }

    attach_workspace_image(
        relationship_prompt,
        figures_dir / "relationship-preview.png",
        "主要因素与结果的原始关系预览图",
        "请先观察图中的整体方向或类别分布，再选择最符合你判断的选项。",
    )

    plot_terms = results_df[
        (results_df["term"] != "const")
        & (results_df.get("term_type", "coefficient") == "coefficient")
        & ~results_df["continuous_functional_form"].isin(
            ["quadratic", "restricted-cubic-spline"]
        )
    ].copy()
    if not plot_terms.empty:
        plot_terms["plot_label"] = plot_terms.apply(
            lambda row: (
                f"{row['outcome_category']}：{row['term']}"
                if row.get("outcome_category")
                else str(row["term"])
            ),
            axis=1,
        )
        fig, ax = plt.subplots(figsize=(10, max(5.2, 0.6 * len(plot_terms))))
        y_pos = np.arange(len(plot_terms))
        ax.errorbar(
            plot_terms["estimate"],
            y_pos,
            xerr=[
                plot_terms["estimate"] - plot_terms["ci_low"],
                plot_terms["ci_high"] - plot_terms["estimate"],
            ],
            fmt="o",
            color=PRIMARY_BLUE_GRAY,
            ecolor=SECONDARY_BLUE_GRAY,
            markersize=7,
            elinewidth=1.8,
            capsize=4,
            capthick=1.4,
        )
        ax.axvline(0, color=DARK_BLUE_GRAY, linewidth=1.1)
        ax.set_yticks(y_pos, plot_terms["plot_label"])
        ax.invert_yaxis()
        ax.set_xlabel(f"系数估计值（{confidence:.0%}置信区间）")
        ax.set_title("模型系数与置信区间", fontweight="bold", pad=14)
        polish_axes(ax, grid_axis="x")
        save_figure(fig, figures_dir / "coefficients.png")

    nonlinear_curve_figures: list[str] = []
    if model_type in SUPPORTED_MODEL_TYPE_SET:
        parameter_vector: np.ndarray | None = None
        covariance_matrix: np.ndarray | None = None
        if model_type in {"ols", "logistic", "poisson", "negative-binomial"}:
            fitted_term_count = len(x.columns)
            parameter_vector = np.asarray(result.params, dtype=float).reshape(-1)[:fitted_term_count]
            covariance_matrix = np.asarray(result.cov_params(), dtype=float)[
                :fitted_term_count, :fitted_term_count
            ]
        critical_value = (
            float(stats.t.ppf(0.5 + confidence / 2, max(float(result.df_resid), 1.0)))
            if model_type == "ols"
            else float(stats.norm.ppf(0.5 + confidence / 2))
        )
        for column, form_spec in sorted(continuous_form_specs.items()):
            if form_spec["form"] == "linear":
                continue
            raw_values = pd.to_numeric(frame[column], errors="coerce")
            grid = np.linspace(
                float(raw_values.quantile(0.02)),
                float(raw_values.quantile(0.98)),
                120,
            )
            representative = pd.DataFrame(index=range(len(grid)))
            for variable in [*predictors, *controls]:
                if variable == column:
                    representative[variable] = grid
                elif variable in categorical_predictors:
                    representative[variable] = categorical_references[variable]
                else:
                    representative[variable] = float(
                        pd.to_numeric(frame[variable], errors="coerce").median()
                    )
            curve_frame = apply_continuous_forms(
                representative, continuous_form_specs
            )
            for factor in sorted(categorical_predictors):
                observed = sorted(frame[factor].astype(str).unique().tolist())
                reference = categorical_references[factor]
                categories = [reference, *(value for value in observed if value != reference)]
                curve_frame[factor] = pd.Categorical(
                    curve_frame[factor].astype(str), categories=categories, ordered=False
                )
            curve_design = pd.get_dummies(
                curve_frame,
                columns=sorted(categorical_predictors),
                drop_first=True,
                dtype=float,
            ).reindex(columns=x_no_constant.columns, fill_value=0.0)
            curve_design_no_constant = curve_design.astype(float)

            if model_type in {"multinomial-logistic", "ordinal-logistic"}:
                if not outcome_categories:
                    fail("Adjusted category-probability curve is missing outcome categories")
                if model_type == "multinomial-logistic":
                    curve_design_for_prediction = sm.add_constant(
                        curve_design_no_constant, has_constant="add"
                    )
                    if list(map(str, curve_design_for_prediction.columns)) != list(map(str, x.columns)):
                        fail(f"Adjusted multinomial curve design drifted for {column!r}")
                    prediction_matrix = curve_design_for_prediction.to_numpy(dtype=float)
                    parameter_matrix = np.asarray(result.params, dtype=float)

                    def predict_categories(parameters: np.ndarray) -> np.ndarray:
                        coefficients = np.asarray(parameters, dtype=float).reshape(
                            parameter_matrix.shape, order="F"
                        )
                        nonreference_logits = prediction_matrix @ coefficients
                        logits = np.column_stack(
                            [np.zeros(len(prediction_matrix)), nonreference_logits]
                        )
                        logits -= np.max(logits, axis=1, keepdims=True)
                        exponentiated = np.exp(logits)
                        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

                    flat_parameters = parameter_matrix.reshape(-1, order="F")
                else:
                    if list(map(str, curve_design_no_constant.columns)) != list(
                        map(str, x_no_constant.columns)
                    ):
                        fail(f"Adjusted ordinal curve design drifted for {column!r}")

                    def predict_categories(parameters: np.ndarray) -> np.ndarray:
                        return np.asarray(
                            result.model.predict(
                                np.asarray(parameters, dtype=float),
                                exog=curve_design_no_constant,
                            ),
                            dtype=float,
                        )

                    flat_parameters = np.asarray(result.params, dtype=float).reshape(-1)

                full_covariance = np.asarray(result.cov_params(), dtype=float)
                if full_covariance.shape != (len(flat_parameters), len(flat_parameters)):
                    fail(f"Adjusted category-probability covariance drifted for {column!r}")
                symmetric_covariance = (full_covariance + full_covariance.T) / 2.0
                eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
                positive_semidefinite_covariance = (
                    eigenvectors
                    @ np.diag(np.maximum(eigenvalues, 0.0))
                    @ eigenvectors.T
                )
                rng = np.random.default_rng(20260818)
                parameter_draws = rng.multivariate_normal(
                    flat_parameters,
                    positive_semidefinite_covariance,
                    size=300,
                    check_valid="ignore",
                )
                predicted = predict_categories(flat_parameters)
                if model_type == "multinomial-logistic":
                    statsmodels_prediction = np.asarray(
                        result.model.predict(
                            result.params,
                            exog=curve_design_for_prediction,
                        ),
                        dtype=float,
                    )
                    if not np.allclose(predicted, statsmodels_prediction, rtol=1e-9, atol=1e-11):
                        fail(
                            f"Adjusted multinomial category order drifted for {column!r}"
                        )
                simulated = np.stack(
                    [predict_categories(draw) for draw in parameter_draws], axis=0
                )
                tail_probability = (1.0 - confidence) / 2.0
                lower = np.quantile(simulated, tail_probability, axis=0)
                upper = np.quantile(simulated, 1.0 - tail_probability, axis=0)
                if predicted.shape != (len(grid), len(outcome_categories)):
                    fail(f"Adjusted category probabilities have the wrong shape for {column!r}")
                fig, ax = plt.subplots(figsize=(9.2, 6.2))
                palette = plt.get_cmap("tab10")
                for category_index, category in enumerate(outcome_categories):
                    color = palette(category_index % 10)
                    ax.plot(
                        grid,
                        predicted[:, category_index],
                        color=color,
                        linewidth=2.3,
                        label=str(category),
                    )
                    ax.fill_between(
                        grid,
                        lower[:, category_index],
                        upper[:, category_index],
                        color=color,
                        alpha=0.14,
                    )
                figure_name = (
                    f"adjusted-category-probabilities-{safe_figure_stem(column)}.png"
                )
                ax.set_xlabel(column)
                ax.set_ylabel("调整后预测概率")
                ax.set_ylim(0, 1)
                ax.set_yticks(
                    np.linspace(0, 1, 6),
                    [f"{value:.0%}" for value in np.linspace(0, 1, 6)],
                )
                ax.set_title(
                    (
                        f"{column}与各类别概率的调整后关系"
                        if model_type == "multinomial-logistic"
                        else f"{column}与各等级概率的调整后关系"
                    ),
                    fontweight="bold",
                    pad=14,
                )
                ax.legend(
                    title=("结果类别" if model_type == "multinomial-logistic" else "结果等级"),
                    frameon=False,
                    ncol=min(3, len(outcome_categories)),
                )
                polish_axes(ax, grid_axis="y")
                save_figure(fig, figures_dir / figure_name)
                nonlinear_curve_figures.append(figure_name)
                continue

            curve_design_with_constant = sm.add_constant(
                curve_design_no_constant, has_constant="add"
            )
            if list(map(str, curve_design_with_constant.columns)) != list(map(str, x.columns)):
                fail(f"Adjusted curve design drifted for {column!r}")
            matrix = curve_design_with_constant.to_numpy(dtype=float)
            assert parameter_vector is not None and covariance_matrix is not None
            linear_predictor = matrix @ parameter_vector
            standard_error = np.sqrt(
                np.maximum(np.einsum("ij,jk,ik->i", matrix, covariance_matrix, matrix), 0.0)
            )
            fig, ax = plt.subplots(figsize=(8.6, 5.9))
            if model_type == "logistic":
                predicted = 1.0 / (1.0 + np.exp(-linear_predictor))
                lower = 1.0 / (
                    1.0 + np.exp(-(linear_predictor - critical_value * standard_error))
                )
                upper = 1.0 / (
                    1.0 + np.exp(-(linear_predictor + critical_value * standard_error))
                )
                figure_name = f"adjusted-probability-{safe_figure_stem(column)}.png"
                curve_label = "调整后概率"
                y_label = f"{outcome}={spec.get('positive_class')} 的预测概率"
                title = f"{column}与结果概率的调整后关系"
                ax.set_ylim(0, 1)
                ax.set_yticks(
                    np.linspace(0, 1, 6),
                    [f"{value:.0%}" for value in np.linspace(0, 1, 6)],
                )
            elif model_type == "ols":
                predicted = linear_predictor
                lower = linear_predictor - critical_value * standard_error
                upper = linear_predictor + critical_value * standard_error
                figure_name = f"adjusted-outcome-{safe_figure_stem(column)}.png"
                curve_label = "调整后预测结果"
                y_label = f"{outcome} 的预测值"
                title = f"{column}与结果的调整后关系"
            else:
                predicted = np.exp(np.clip(linear_predictor, -700, 700))
                lower = np.exp(
                    np.clip(linear_predictor - critical_value * standard_error, -700, 700)
                )
                upper = np.exp(
                    np.clip(linear_predictor + critical_value * standard_error, -700, 700)
                )
                figure_name = f"adjusted-count-{safe_figure_stem(column)}.png"
                curve_label = "调整后预计计数"
                y_label = f"{outcome} 的预计计数"
                title = f"{column}与预计计数的调整后关系"
            ax.plot(
                grid,
                predicted,
                color=PRIMARY_BLUE_GRAY,
                linewidth=2.5,
                label=curve_label,
            )
            ax.fill_between(
                grid,
                lower,
                upper,
                color=SECONDARY_BLUE_GRAY,
                alpha=0.28,
                label=f"{confidence:.0%}置信带",
            )
            ax.set_xlabel(column)
            ax.set_ylabel(y_label)
            ax.set_title(title, fontweight="bold", pad=14)
            ax.legend(frameon=False)
            polish_axes(ax, grid_axis="y")
            save_figure(fig, figures_dir / figure_name)
            nonlinear_curve_figures.append(figure_name)

    diagnostic_figure_name: str
    if model_type == "ols":
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        ax.scatter(fitted, resid, s=52, alpha=0.78, color=PRIMARY_BLUE_GRAY, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color=DARK_BLUE_GRAY, linewidth=1.1)
        ax.set_xlabel("拟合值")
        ax.set_ylabel("OLS 残差")
        ax.set_title("OLS：残差与拟合值", fontweight="bold", pad=14)
        polish_axes(ax, grid_axis="y")
        save_figure(fig, figures_dir / "residuals-vs-fitted.png")

        fig, ax = plt.subplots(figsize=(7.2, 7.0))
        (theoretical_quantiles, ordered_residuals), (qq_slope, qq_intercept, _qq_r) = stats.probplot(resid, dist="norm")
        ax.scatter(theoretical_quantiles, ordered_residuals, s=46, alpha=0.8, color=PRIMARY_BLUE_GRAY, edgecolor="white", linewidth=0.5)
        qq_x = np.asarray([min(theoretical_quantiles), max(theoretical_quantiles)])
        ax.plot(qq_x, qq_intercept + qq_slope * qq_x, color=SECONDARY_BLUE_GRAY, linewidth=2)
        ax.set_xlabel("理论分位数")
        ax.set_ylabel("样本残差分位数")
        ax.set_title("OLS：残差 Q-Q 图", fontweight="bold", pad=14)
        polish_axes(ax)
        save_figure(fig, figures_dir / "residual-qq.png")
        diagnostic_figure_name = "residuals-vs-fitted.png"
    elif model_type == "logistic":
        assert y is not None
        y_array = np.asarray(y, dtype=int)
        false_positive, true_positive = binary_roc_points(y_array, fitted)
        fig, ax = plt.subplots(figsize=(7.6, 6.5))
        ax.plot(false_positive, true_positive, color=PRIMARY_BLUE_GRAY, linewidth=2.4, label=f"AUC={metrics.get('roc_auc'):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#87979e", label="随机参考")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="假阳性率", ylabel="真阳性率", title="Logistic：ROC 曲线")
        ax.legend(frameon=False)
        polish_axes(ax, grid_axis="both")
        save_figure(fig, figures_dir / "logistic-roc.png")

        calibration = calibration_table(y_array, fitted)
        fig, ax = plt.subplots(figsize=(7.6, 6.5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#87979e", label="理想校准")
        ax.plot(calibration["predicted"], calibration["observed"], marker="o", color=PRIMARY_BLUE_GRAY, linewidth=2.2, label=("折外预测" if cross_validated_scalar_predictions is not None else "样本内拟合"))
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="平均预测概率", ylabel="实际正类比例", title="Logistic：校准图")
        ax.legend(frameon=False)
        polish_axes(ax, grid_axis="both")
        save_figure(fig, figures_dir / "logistic-calibration.png")
        diagnostic_figure_name = "logistic-calibration.png"
    elif model_type in {"poisson", "negative-binomial"}:
        assert y is not None
        observed_counts = np.asarray(y, dtype=float)
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.scatter(fitted, resid, s=48, alpha=0.72, color=PRIMARY_BLUE_GRAY, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=DARK_BLUE_GRAY, linewidth=1.1)
        ax.set_xlabel("拟合计数")
        ax.set_ylabel("模型残差")
        ax.set_title(f"{model_type}：残差与拟合计数", fontweight="bold", pad=14)
        polish_axes(ax, grid_axis="y")
        save_figure(fig, figures_dir / "count-residuals-vs-fitted.png")

        fig, ax = plt.subplots(figsize=(7.4, 6.4))
        ax.scatter(fitted, observed_counts, s=48, alpha=0.72, color=PRIMARY_BLUE_GRAY, edgecolor="white", linewidth=0.5)
        upper = max(float(np.max(fitted)), float(np.max(observed_counts)), 1.0)
        ax.plot([0, upper], [0, upper], linestyle="--", color="#87979e")
        ax.set(xlabel="拟合计数", ylabel="实际计数", title=f"{model_type}：实际计数与拟合计数")
        polish_axes(ax, grid_axis="both")
        save_figure(fig, figures_dir / "count-observed-vs-fitted.png")
        diagnostic_figure_name = "count-residuals-vs-fitted.png"
    else:
        assert y is not None and fitted_probabilities is not None
        actual_codes = np.asarray(y, dtype=int)
        predicted_codes = np.argmax(fitted_probabilities, axis=1)
        confusion = pd.crosstab(actual_codes, predicted_codes).reindex(
            index=range(len(outcome_categories)), columns=range(len(outcome_categories)), fill_value=0
        )
        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        image = ax.imshow(confusion.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(outcome_categories)), outcome_categories, rotation=25, ha="right")
        ax.set_yticks(range(len(outcome_categories)), outcome_categories)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("实际类别")
        ax.set_title("多分类混淆矩阵" if model_type == "multinomial-logistic" else "有序分类混淆矩阵")
        for row_index in range(len(outcome_categories)):
            for column_index in range(len(outcome_categories)):
                ax.text(column_index, row_index, str(int(confusion.iat[row_index, column_index])), ha="center", va="center")
        fig.colorbar(image, ax=ax, label="记录数")
        save_figure(fig, figures_dir / "classification-confusion.png")

        fig, ax = plt.subplots(figsize=(8.8, 6.5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#87979e", label="理想校准")
        palette = plt.get_cmap("tab10")
        calibration_probabilities = (
            np.asarray(cross_validated_probability_predictions, dtype=float)
            if cross_validated_probability_predictions is not None
            else fitted_probabilities
        )
        if model_type == "multinomial-logistic":
            for category_index, category in enumerate(outcome_categories):
                table = calibration_table(
                    (actual_codes == category_index).astype(float),
                    calibration_probabilities[:, category_index],
                )
                if table.empty:
                    continue
                ax.plot(
                    table["predicted"],
                    table["observed"],
                    marker="o",
                    linewidth=2.0,
                    color=palette(category_index % 10),
                    label=f"类别：{category}",
                )
            calibration_title = "多分类：各类别概率校准"
            calibration_note = "每条线按该类别的一对其余类别概率分组"
        else:
            for threshold in range(len(outcome_categories) - 1):
                table = calibration_table(
                    (actual_codes <= threshold).astype(float),
                    calibration_probabilities[:, : threshold + 1].sum(axis=1),
                )
                if table.empty:
                    continue
                ax.plot(
                    table["predicted"],
                    table["observed"],
                    marker="o",
                    linewidth=2.0,
                    color=palette(threshold % 10),
                    label=f"累计至：{outcome_categories[threshold]}",
                )
            calibration_title = "有序分类：各等级分界累计概率校准"
            calibration_note = "每条线对应一个从最低等级累计到当前等级的分界"
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="分组平均预测概率", ylabel="分组实际发生率")
        ax.set_title(calibration_title, fontweight="bold", pad=14)
        ax.text(0.02, 0.98, calibration_note, transform=ax.transAxes, va="top", fontsize=9, color=DARK_BLUE_GRAY)
        ax.legend(frameon=False, fontsize=9)
        polish_axes(ax, grid_axis="both")
        save_figure(fig, figures_dir / "classification-calibration.png")
        diagnostic_figure_name = "classification-calibration.png"

    if influence.get("status") == "available":
        influence_scores = np.asarray(influence["priority_score"], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5.4))
        markerline, stemlines, baseline = ax.stem(
            np.arange(1, len(influence_scores) + 1), influence_scores, basefmt=" "
        )
        plt.setp(markerline, color=PRIMARY_BLUE_GRAY, markersize=5)
        plt.setp(stemlines, color=SECONDARY_BLUE_GRAY, linewidth=1.2)
        baseline.set_visible(False)
        ax.axhline(
            1.0,
            color=DARK_BLUE_GRAY,
            linestyle="--",
            linewidth=1.2,
            label="候选筛查参考线",
        )
        ax.set_xlabel("清洗数据中的观测序号")
        ax.set_ylabel("模型特定影响优先级得分")
        ax.set_title(f"{model_type}：高影响记录筛查", fontweight="bold", pad=14)
        ax.legend()
        polish_axes(ax, grid_axis="y")
        save_figure(fig, figures_dir / "influence.png")

    bp_value = metrics.get("breusch_pagan_p_value")
    dispersion_value = metrics.get("dispersion")
    if model_type == "ols" and bp_value is not None:
        diagnostic_evidence = (
            f"本次 Breusch–Pagan 检验 p={float(bp_value):.3g}。"
            + (
                "没有显示明显异方差证据，但仍要结合残差图和样本量判断。"
                if float(bp_value) >= 0.05
                else "提示残差方差可能不稳定，应优先查看稳健标准误和敏感性分析。"
            )
        )
        diagnostic_prompt = {
            "question": "OLS 残差的波动是否随着拟合值增大而系统变化？",
            "options": ["可能变化，需要检查方差假设", "大致稳定", "仅凭图无法判断"],
            "evidence_answer": "仅凭图无法判断",
            "evidence_feedback": diagnostic_evidence,
        }
    elif model_type in {"poisson", "negative-binomial"}:
        diagnostic_evidence = (
            f"本次计数模型离散度为 {float(dispersion_value):.3g}。"
            + (
                "明显高于 1，说明 Poisson 方差假设可能不足。"
                if dispersion_value is not None and float(dispersion_value) > 1.5
                else "接近 1，未见明显过度离散。"
            )
        ) if dispersion_value is not None else "请检查计数残差是否围绕零线随机分布，以及实际计数与拟合计数是否系统偏离。"
        diagnostic_prompt = {
            "question": "计数模型是否出现系统残差或明显过度离散？",
            "options": ["存在明显问题", "未见明显问题", "仅凭图无法判断"],
            "evidence_answer": "仅凭图无法判断",
            "evidence_feedback": diagnostic_evidence,
        }
    elif model_type == "logistic":
        diagnostic_prompt = {
            "question": "校准图中的实际正类比例是否接近理想对角线？",
            "options": ["较为接近", "偏离明显", "仅凭图无法判断"],
            "evidence_answer": "仅凭图无法判断",
            "evidence_feedback": (
                f"本次 AUC={metrics.get('roc_auc'):.3f}，Brier 分数={metrics.get('brier_score'):.3f}。"
                "AUC反映区分能力，校准图和Brier分数反映概率是否可信，两者不能互相替代。"
            ),
        }
    elif model_type == "multinomial-logistic":
        diagnostic_prompt = {
            "question": "混淆矩阵是否显示某些结果类别经常被错分？",
            "options": ["存在集中错分", "各类别识别较均衡", "仅凭图无法判断"],
            "evidence_answer": "仅凭图无法判断",
            "evidence_feedback": "应同时查看混淆矩阵、逐类别概率校准曲线和多分类对数损失，不能只看总体准确率；预测任务优先解释折外校准。",
        }
    else:
        diagnostic_prompt = {
            "question": "有序分类的错分是否大多发生在相邻等级，而不是跨越多个等级？",
            "options": ["主要是相邻等级错分", "存在较多跨级错分", "仅凭图无法判断"],
            "evidence_answer": "仅凭图无法判断",
            "evidence_feedback": (
                "应结合有序混淆矩阵、平均等级误差和各等级分界的累计概率校准；"
                f"本次比例优势等斜率检查p={runtime_proportional_odds.get('p_value'):.3g}，"
                "未拒绝只表示当前样本未发现明显违背证据。"
            ),
        }
    attach_workspace_image(
        diagnostic_prompt,
        figures_dir / diagnostic_figure_name,
        f"{model_type} 模型专用诊断图",
        "该图只用于当前模型对应的诊断问题，不套用其他模型的残差假设。",
    )
    learning_prompts = build_learning_prompts(
        relationship_prompt,
        diagnostic_prompt,
        collinearity_summary,
        severe_vif_terms,
        influence_summary,
    )

    write_analysis_tables(
        output_dir,
        results_df,
        diagnostics,
        factor_tests,
        continuous_shape_tests,
        influence_rows,
    )
    write_json_artifacts(
        output_dir,
        {
            "proportional-odds-check.json": runtime_proportional_odds,
            "iia-check.json": runtime_iia,
            "count-dispersion-check.json": runtime_count_dispersion,
            "negative-binomial-need-check.json": runtime_negative_binomial_need,
            "zero-inflation-check.json": runtime_zero_inflation,
            "predictive-validation.json": predictive_validation,
            "learning-prompts.json": learning_prompts,
        },
    )
    warnings = list(spec.get("warnings", []))
    warnings.extend(
        [
            "P-values must be interpreted together with effect sizes and confidence intervals.",
            "The fitted model estimates statistical associations, not causal effects.",
        ]
    )
    if warnings_for_prediction:
        warnings.append(warnings_for_prediction)
    if severe_vif_terms:
        warnings.append(
            "Severe multicollinearity (VIF >= 10) restricts individual-effect "
            "interpretation for: " + ", ".join(severe_vif_terms)
        )
    if missingness_screen.get("status") == "review-required":
        warnings.append(
            "Selected variables had missing values before preparation. The missingness screen is descriptive only; "
            "because model-estimate sensitivity was not completed, interpretations apply only to the analyzed sample."
        )
    if influence_summary["status"] == "not-evaluated":
        warnings.append(
            "Model-specific influence diagnostics were not evaluated; this must not be interpreted as zero influential records."
        )
    elif influence_summary.get("sensitivity_status") in {"partial", "not-evaluated"}:
        warnings.append(
            "Some candidate case-deletion refits failed, so influence sensitivity evidence is incomplete."
        )
    if int(influence_summary.get("sign_flip_count") or 0) > 0 or int(
        influence_summary.get("significance_flip_count") or 0
    ) > 0:
        warnings.append(
            "At least one evaluated candidate changed a coefficient sign or significance decision when omitted; retain the original fit but restrict strong conclusions."
        )
    summary = {
        "status": "completed",
        "modeling_executed": True,
        "model_type": model_type,
        "outcome": outcome,
        "predictors": predictors,
        "controls": controls,
        "outcome_categories": outcome_categories or None,
        "positive_class": spec.get("positive_class"),
        "outcome_reference_class": spec.get("outcome_reference_class"),
        "reference_class": spec.get("reference_class"),
        "categorical_reference_categories": categorical_references,
        "category_support_screen": runtime_category_support,
        "iia_check": runtime_iia,
        "proportional_odds_check": runtime_proportional_odds,
        "count_dispersion_check": runtime_count_dispersion,
        "negative_binomial_need_check": runtime_negative_binomial_need,
        "zero_inflation_check": runtime_zero_inflation,
        "predictive_validation": predictive_validation,
        "continuous_functional_forms": continuous_form_specs,
        "continuous_shape_tests": continuous_shape_tests,
        "nonlinear_curve_figures": nonlinear_curve_figures,
        "nonlinear_probability_figures": (
            nonlinear_curve_figures if model_type == "logistic" else []
        ),
        "nonlinear_category_probability_figures": (
            nonlinear_curve_figures
            if model_type in {"multinomial-logistic", "ordinal-logistic"}
            else []
        ),
        "collinearity": collinearity_summary,
        "multiplicity": {
            "coefficient_method": "Benjamini-Hochberg FDR",
            "coefficient_family": "all non-intercept coefficients",
            "coefficient_tests": len(coefficient_row_indices),
            "categorical_omnibus_method": "joint Wald chi-square",
            "categorical_omnibus_adjustment": "Benjamini-Hochberg FDR",
            "categorical_omnibus_tests": factor_tests,
            "decision_rule": (
                "A categorical level is highlighted only when both its adjusted "
                "coefficient p-value and its factor's adjusted omnibus p-value "
                f"are below alpha={alpha:.6g}."
            ),
        },
        "rows_used": len(frame),
        "excluded_cleaned_data_rows": excluded_rows,
        "missingness_bias_screen": missingness_screen,
        "missingness_conclusion_contract": missingness_contract,
        "model_complete_case_excluded_row_count": len(excluded_rows),
        "metrics": metrics,
        "diagnostic_applicability": applicability,
        "diagnostic_figures": sorted(
            path.name
            for path in figures_dir.glob("*.png")
            if path.name not in {"relationship-preview.png", "coefficients.png"}
        ),
        "influence_available": influence_summary["status"] == "available",
        "influential_cleaned_data_rows": influential_rows,
        "influence_diagnostics": influence_summary,
        "warnings": warnings,
        "figures": sorted(path.name for path in figures_dir.glob("*.png")),
    }
    write_json_artifacts(output_dir, {"model-summary.json": summary})
    run_log = {
        "status": "completed",
        "executed_at": datetime.now().astimezone().isoformat(),
        "data_sha256": sha256_file(data_path),
        "model_specification_sha256": sha256_file(spec_path),
        "preparation_log_sha256": sha256_file(prep_path),
        "software": {"python": sys.version.split()[0], "pandas": pd.__version__, "statsmodels": statsmodels_package.__version__},
        "outputs": [path.name for path in targets] + [f"figures/{name}" for name in summary["figures"]],
    }
    write_json_artifacts(output_dir, {"analysis-run-log.json": run_log})
    summary_md = "\n".join(
        [
            "# 统计建模结果摘要",
            "",
            f"- 模型：`{model_type}`",
            f"- 因变量：`{outcome}`",
            *(
                [
                    f"- IIA假设检查：`{runtime_iia.get('status')}`；最小Holm校正p值={scalar(runtime_iia.get('minimum_adjusted_p_value'))}",
                    "  - 未发现敏感性只表示当前样本未发现明显违背证据，不证明IIA绝对成立；还需结合类别是否天然相似或嵌套理解。",
                ]
                if model_type == "multinomial-logistic"
                else []
            ),
            *(
                [
                    f"- 二分类目标事件：`{spec.get('positive_class')}`",
                    f"- 二分类结果基准类别：`{spec.get('outcome_reference_class')}`",
                ]
                if model_type == "logistic"
                else []
            ),
            *(
                [f"- 多分类结果参照类别：`{spec.get('reference_class')}`"]
                if model_type == "multinomial-logistic"
                else []
            ),
            *(
                ["- 分类自变量参照组："]
                + [
                    f"  - `{column}`：`{categorical_references[column]}`"
                    for column in sorted(categorical_references)
                ]
                if categorical_references
                else []
            ),
            f"- 多重比较：对 {len(coefficient_row_indices)} 个非截距系数使用 Benjamini–Hochberg FDR 校正",
            f"- 分类变量总体检验：{len(factor_tests)} 个联合 Wald 检验，并对总体检验p值使用 Benjamini–Hochberg FDR 校正",
            f"- 类别支持度检查：`{runtime_category_support.get('status')}`；模型按已批准的类别集合重新核验",
            *(
                [
                    f"- 比例优势假设检查：`{runtime_proportional_odds.get('status')}`；等斜率Wald检验p值={scalar(runtime_proportional_odds.get('p_value'))}",
                    "  - 未拒绝只表示当前样本未发现明显违背证据，不证明假设绝对成立。",
                ]
                if model_type == "ordinal-logistic"
                else []
            ),
            *(
                [
                    f"- Poisson过度离散检查：`{runtime_count_dispersion.get('status')}`；单侧检验p值={scalar(runtime_count_dispersion.get('p_value'))}",
                    f"  - Pearson离散度={scalar(runtime_count_dispersion.get('pearson_dispersion_descriptive'))}，仅作描述，不按固定阈值自动换模。",
                    "  - 未拒绝只表示当前样本未发现明确的调整后过度离散证据，不证明Poisson方差假设绝对成立。",
                ]
                if model_type == "poisson"
                else []
            ),
            *(
                [
                    f"- 负二项必要性检查：`{runtime_negative_binomial_need.get('status')}`；单侧检验p值={scalar(runtime_negative_binomial_need.get('p_value'))}",
                    "  - 只有检出调整后的额外离散证据时才保留负二项；系统不会自动改成Poisson。",
                ]
                if model_type == "negative-binomial"
                else []
            ),
            *(
                [
                    f"- 过多零值检查：`{runtime_zero_inflation.get('status')}`；实际零值={runtime_zero_inflation.get('observed_zero_count')}；预计零值={scalar(runtime_zero_inflation.get('expected_zero_count'))}",
                    "  - 未检出只表示普通计数模型能解释当前零值频率，不能证明不存在结构性零值。",
                ]
                if model_type in {"poisson", "negative-binomial"}
                else []
            ),
            f"- 预测验证：`{predictive_validation.get('status')}`；{predictive_validation.get('method', predictive_validation.get('reason', '不适用'))}",
            *(
                ["- 连续变量函数形式："]
                + [
                    f"  - `{column}`：{continuous_form_specs[column]['plain_label']}；来源={continuous_form_specs[column]['selection_source']}；理由={continuous_form_specs[column]['selection_rationale']}"
                    for column in sorted(continuous_form_specs)
                ]
                if continuous_form_specs
                else []
            ),
            *(
                [f"- 非线性整体检验：{len(nonlinear_test_items)} 项；非线性基函数只作联合检验，不逐项解释为现实效应"]
                if nonlinear_test_items
                else []
            ),
            f"- 共线性状态：`{collinearity_summary['status']}`；最大VIF={scalar(collinearity_summary['maximum_vif'])}",
            *(
                [
                    "- 单项系数解释限制：以下模型项VIF≥10，不作为稳定独立作用解释：",
                    "  - " + "、".join(f"`{term}`" for term in severe_vif_terms),
                ]
                if severe_vif_terms
                else []
            ),
            f"- 使用样本数：{len(frame)}",
            f"- 排除的清洗数据行数：{len(excluded_rows)}",
            *(
                [
                    f"- 高影响诊断：已评估；候选记录 {len(influential_rows)} 条，逐条删一重拟合 {influence_summary.get('case_deletion_evaluated_count', 0)} 条",
                    f"- 自动删除记录：否；原模型始终保留",
                ]
                if influence_summary["status"] == "available"
                else [f"- 高影响诊断：未能评估（{influence_summary.get('reason', '未记录原因')}）"]
            ),
            "",
            "## 解释边界",
            "",
            "结果必须结合效应方向、效应大小、置信区间和模型诊断解释，不能只依据 p 值，也不能把统计关联表述为因果关系。",
            "",
        ]
    )
    (output_dir / "analysis-summary.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps({"ok": True, "status": "completed", "model_type": model_type, "rows_used": len(frame), "terms": len(results_df), "figures": len(summary["figures"]), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
