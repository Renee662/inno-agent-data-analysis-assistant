#!/usr/bin/env python3
"""Create a reviewable model-specification proposal without fitting a model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from category_support import flatten_category_support_rows, screen_category_support  # noqa: E402
from file_utils import atomic_replace_text, sha256_file  # noqa: E402
from model_design import apply_continuous_forms  # noqa: E402
from model_registry import SUPPORTED_MODEL_TYPES, SUPPORTED_MODEL_TYPE_SET  # noqa: E402
from count_dispersion import (  # noqa: E402
    not_applicable as count_dispersion_not_applicable,
    screen_count_dispersion,
)
from iia import not_applicable as iia_not_applicable, screen_iia  # noqa: E402
from negative_binomial_need import (  # noqa: E402
    not_applicable as negative_binomial_need_not_applicable,
    screen_negative_binomial_need,
)
from proportional_odds import not_applicable, screen_proportional_odds  # noqa: E402
from zero_inflation import (  # noqa: E402
    not_applicable as zero_inflation_not_applicable,
    screen_zero_inflation,
)


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an exact statistical model proposal for user review.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--workflow-support", required=True)
    parser.add_argument("--preparation-log", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
    )
    parser.add_argument("--robust-se", default="HC3", choices=["HC3", "nonrobust"])
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--positive-class")
    parser.add_argument("--reference-class")
    parser.add_argument(
        "--categorical-reference",
        action="append",
        default=[],
        metavar="COLUMN=CATEGORY",
        help="Reference category for a categorical predictor/control; repeat as needed.",
    )
    parser.add_argument("--category-order", nargs="+")
    parser.add_argument(
        "--continuous-form",
        action="append",
        default=[],
        metavar="COLUMN=FORM",
        help=(
            "Optional approved domain override for a continuous predictor; normally omit so the deterministic policy decides. "
            "FORM is linear, quadratic, or restricted-cubic-spline. Repeat as needed."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        fail(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Cannot read {label}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object")
    return value


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    fail(f"Cannot decode cleaned CSV: {path}")
    raise AssertionError("unreachable")


def task_variable_metadata(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = task.get("variable_metadata", {})
    if isinstance(raw, dict):
        return {
            str(column): value
            for column, value in raw.items()
            if isinstance(value, dict)
        }
    if isinstance(raw, list):
        return {
            str(item.get("column")): item
            for item in raw
            if isinstance(item, dict) and item.get("column")
        }
    return {}


def parse_categorical_references(items: list[str]) -> dict[str, str]:
    references: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            fail("categorical-reference must use COLUMN=CATEGORY")
        column, category = item.split("=", 1)
        column = column.strip()
        category = category.strip()
        if not column or not category:
            fail("categorical-reference must contain a non-empty column and category")
        if column in references:
            fail(f"categorical-reference was supplied more than once for {column!r}")
        references[column] = category
    return references


CONTINUOUS_FORMS = {"linear", "quadratic", "restricted-cubic-spline"}
CONTINUOUS_FORM_SOURCE_LABELS = {
    "explicit-approved-domain-override": "已批准的专业设定",
    "approved-task-metadata": "已批准任务中的专业设定",
    "automatic-fixed-flexible-default": "系统固定的灵活默认规则",
    "automatic-limited-support-linear": "取值支持不足时的保守直线规则",
}


def parse_continuous_forms(items: list[str]) -> dict[str, str]:
    forms: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            fail("continuous-form must use COLUMN=FORM")
        column, form = (part.strip() for part in item.split("=", 1))
        if not column or form not in CONTINUOUS_FORMS:
            fail(
                "continuous-form requires a non-empty column and one of: "
                + ", ".join(sorted(CONTINUOUS_FORMS))
            )
        if column in forms:
            fail(f"continuous-form was supplied more than once for {column!r}")
        forms[column] = form
    return forms


def resolve_continuous_forms(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    requested: dict[str, str],
    metadata: dict[str, dict[str, Any]],
    model_type: str,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    if model_type not in SUPPORTED_MODEL_TYPE_SET:
        if requested:
            fail("continuous-form is not supported for this model type")
        return {}, {}
    forms: dict[str, str] = {}
    decisions: dict[str, dict[str, str]] = {}
    for column in continuous_columns:
        item = metadata.get(column, {})
        metadata_form = item.get("functional_form")
        metadata_rationale = str(item.get("functional_form_rationale") or "").strip()
        if column in requested:
            if metadata_form != requested[column] or not metadata_rationale:
                fail(
                    f"continuous-form override for {column!r} requires matching approved task metadata "
                    "and a substantive functional_form_rationale"
                )
            form = requested[column]
            source = "explicit-approved-domain-override"
            rationale = metadata_rationale
        elif metadata_form in CONTINUOUS_FORMS and metadata_rationale:
            form = str(metadata_form)
            source = "approved-task-metadata"
            rationale = metadata_rationale
        elif pd.to_numeric(frame[column], errors="coerce").nunique() >= 8:
            form = "restricted-cubic-spline"
            source = "automatic-fixed-flexible-default"
            rationale = "没有已批准的形状先验；取值支持固定4结点样条，因此预先保留可能的弯曲关系。"
        else:
            form = "linear"
            source = "automatic-limited-support-linear"
            rationale = "不同取值少于8个，无法可靠构造固定4结点样条；采用直线并保留非线性未充分评估的限制。"
        forms[column] = form
        decisions[column] = {"source": source, "rationale": rationale}
    return forms, decisions


def continuous_form_specifications(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    forms: dict[str, str],
    outcome: str,
    positive_class: str | None,
    model_type: str = "logistic",
    outcome_categories: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    specifications: dict[str, dict[str, Any]] = {}
    for column in continuous_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or values.nunique() < 2:
            fail(f"Continuous predictor {column!r} must contain at least two numeric values")
        form = forms[column]
        center = float(values.mean())
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            fail(f"Continuous predictor {column!r} has no usable scale")
        item: dict[str, Any] = {
            "form": form,
            "center": center,
            "scale": scale,
            "term_names": [column],
            "plain_label": {
                "linear": "线性项",
                "quadratic": "线性项加二次项",
                "restricted-cubic-spline": "4结点限制性立方样条",
            }[form],
        }
        bin_count = min(8, int(values.nunique()))
        bins = pd.qcut(values, q=bin_count, duplicates="drop")
        if model_type == "logistic":
            if positive_class is None:
                fail("Logistic continuous-form preview requires the confirmed positive class")
            preview = pd.DataFrame(
                {
                    "value": values,
                    "observed": (frame[outcome].astype(str) == positive_class).astype(float),
                    "bin": bins,
                }
            ).groupby("bin", observed=False).agg(
                representative_value=("value", "median"),
                observed_value=("observed", "mean"),
                observations=("observed", "size"),
            )
            item["preview_kind"] = "unadjusted-binned-positive-rate"
            item["preview_value_label"] = "observed_positive_rate"
            item["unadjusted_binned_rate_preview"] = [
                {
                    "representative_value": float(row.representative_value),
                    "observed_positive_rate": float(row.observed_value),
                    "observations": int(row.observations),
                }
                for row in preview.itertuples(index=False)
            ]
            item["unadjusted_binned_preview"] = [
                {
                    "representative_value": point["representative_value"],
                    "observed_value": point["observed_positive_rate"],
                    "observations": point["observations"],
                }
                for point in item["unadjusted_binned_rate_preview"]
            ]
        elif model_type in {"ols", "poisson", "negative-binomial"}:
            numeric_outcome = pd.to_numeric(frame[outcome], errors="coerce")
            if numeric_outcome.isna().any():
                fail("OLS continuous-form preview requires a numeric outcome")
            preview = pd.DataFrame(
                {"value": values, "observed": numeric_outcome, "bin": bins}
            ).groupby("bin", observed=False).agg(
                representative_value=("value", "median"),
                observed_value=("observed", "mean"),
                observed_standard_deviation=("observed", "std"),
                observations=("observed", "size"),
            )
            item["preview_kind"] = (
                "unadjusted-binned-outcome-mean"
                if model_type == "ols"
                else "unadjusted-binned-count-mean"
            )
            item["preview_value_label"] = (
                "observed_outcome_mean" if model_type == "ols" else "observed_count_mean"
            )
            item["unadjusted_binned_preview"] = [
                {
                    "representative_value": float(row.representative_value),
                    "observed_value": float(row.observed_value),
                    "standard_error": (
                        float(row.observed_standard_deviation) / np.sqrt(int(row.observations))
                        if int(row.observations) > 1 and np.isfinite(row.observed_standard_deviation)
                        else None
                    ),
                    "observations": int(row.observations),
                }
                for row in preview.itertuples(index=False)
            ]
        elif model_type in {"multinomial-logistic", "ordinal-logistic"}:
            categories = [str(value) for value in (outcome_categories or [])]
            if len(categories) < 3:
                fail("Multicategory continuous-form preview requires approved outcome categories")
            observed_categories = set(frame[outcome].astype(str).unique())
            if set(categories) != observed_categories:
                fail("Approved outcome categories do not match the continuous-form preview data")
            preview_frame = pd.DataFrame(
                {
                    "value": values,
                    "outcome_category": frame[outcome].astype(str),
                    "bin": bins,
                }
            )
            grouped = preview_frame.groupby("bin", observed=False)
            points: list[dict[str, Any]] = []
            for _bin, group in grouped:
                if group.empty:
                    continue
                counts = group["outcome_category"].value_counts()
                observations = int(len(group))
                points.append(
                    {
                        "representative_value": float(group["value"].median()),
                        "observations": observations,
                        "observed_category_proportions": {
                            category: float(counts.get(category, 0) / observations)
                            for category in categories
                        },
                    }
                )
            item["preview_kind"] = "unadjusted-binned-category-proportions"
            item["preview_value_label"] = "observed_category_proportions"
            item["outcome_category_order"] = categories
            item["unadjusted_binned_category_preview"] = points
        else:
            fail("Continuous functional-form previews are unsupported for this model")
        if form == "quadratic":
            if values.nunique() < 4:
                fail(f"Quadratic form for {column!r} requires at least four distinct values")
            item["term_names"] = [column, f"{column}__quadratic"]
        elif form == "restricted-cubic-spline":
            if values.nunique() < 8:
                fail(
                    f"Restricted cubic spline for {column!r} requires at least eight distinct values"
                )
            standardized = (values - center) / scale
            knots = [float(value) for value in standardized.quantile([0.05, 0.35, 0.65, 0.95])]
            if len(set(round(value, 12) for value in knots)) != 4:
                fail(f"Restricted cubic spline knots are not distinct for {column!r}")
            item["knot_quantiles"] = [0.05, 0.35, 0.65, 0.95]
            item["knots_standardized"] = knots
            item["knots_original_units"] = [center + scale * value for value in knots]
            item["term_names"] = [column, f"{column}__rcs1", f"{column}__rcs2"]
        specifications[column] = item
    return specifications


def format_continuous_preview(item: dict[str, Any]) -> str:
    if item.get("preview_kind") == "unadjusted-binned-positive-rate":
        return "未经调整的分组正类率=" + "、".join(
            f"{point['representative_value']:.4g}:{point['observed_value']:.1%}"
            for point in item.get("unadjusted_binned_preview", [])
        )
    if item.get("preview_kind") in {
        "unadjusted-binned-outcome-mean",
        "unadjusted-binned-count-mean",
    }:
        label = (
            "未经调整的分组结果均值="
            if item.get("preview_kind") == "unadjusted-binned-outcome-mean"
            else "未经调整的分组平均计数="
        )
        return label + "、".join(
            f"{point['representative_value']:.4g}:{point['observed_value']:.4g}"
            for point in item.get("unadjusted_binned_preview", [])
        )
    if item.get("preview_kind") == "unadjusted-binned-category-proportions":
        return "未经调整的分组类别比例=" + "；".join(
            f"{point['representative_value']:.4g}:"
            + "/".join(
                f"{category} {proportion:.1%}"
                for category, proportion in point.get(
                    "observed_category_proportions", {}
                ).items()
            )
            for point in item.get("unadjusted_binned_category_preview", [])
        )
    return "未经调整的分组预览未生成"


def encode_categorical_predictors(
    frame: pd.DataFrame,
    categorical: set[str],
    references: dict[str, str],
) -> pd.DataFrame:
    encoded = frame.copy()
    if set(references) != categorical:
        missing = sorted(categorical - set(references))
        extra = sorted(set(references) - categorical)
        fail(f"Categorical reference map mismatch; missing={missing}, extra={extra}")
    for column in sorted(categorical):
        values = encoded[column].astype(str)
        observed = sorted(values.unique().tolist())
        reference = references[column]
        if reference not in observed:
            fail(f"Reference category {reference!r} is not observed for {column!r}: {observed}")
        categories = [reference, *(value for value in observed if value != reference)]
        encoded[column] = pd.Categorical(values, categories=categories, ordered=False)
    return pd.get_dummies(
        encoded, columns=sorted(categorical), drop_first=True, dtype=float
    )


def design_matrix_preflight(
    encoded_predictors: pd.DataFrame, model_type: str
) -> dict[str, Any]:
    """Validate the exact matrix that the approved runtime will fit."""
    numeric = encoded_predictors.astype(float)
    design = (
        numeric
        if model_type == "ordinal-logistic"
        else sm.add_constant(numeric, has_constant="add")
    )
    values = design.to_numpy(dtype=float)
    column_names = [str(column) for column in design.columns]
    if not np.isfinite(values).all():
        return {
            "status": "revision-required",
            "model_fitting_allowed": False,
            "reason": "non-finite-design-values",
            "row_count": len(design),
            "column_count": len(column_names),
            "rank": None,
            "design_columns": column_names,
            "dependent_columns": [],
            "plain_explanation": "最终设计矩阵含有无法用于拟合的非有限数值，需先修订数据处理或变量形式。",
        }

    rank = int(np.linalg.matrix_rank(values))
    dependent_columns: list[str] = []
    retained = np.empty((len(design), 0), dtype=float)
    retained_rank = 0
    for index, column in enumerate(column_names):
        candidate = np.column_stack([retained, values[:, index]])
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank == retained_rank:
            dependent_columns.append(column)
        else:
            retained = candidate
            retained_rank = candidate_rank
    allowed = rank == len(column_names)
    return {
        "status": "passed" if allowed else "revision-required",
        "model_fitting_allowed": allowed,
        "reason": None if allowed else "rank-deficient-final-design-matrix",
        "row_count": len(design),
        "column_count": len(column_names),
        "rank": rank,
        "design_columns": column_names,
        "dependent_columns": dependent_columns,
        "plain_explanation": (
            "最终设计矩阵已通过满秩与有限值检查，可进入唯一一次模型规格审批。"
            if allowed
            else "最终设计矩阵存在完全重复的信息，当前模型不可识别；需先修订变量、类别编码或函数形式。"
        ),
    }


def screen_collinearity_candidates(
    frame: pd.DataFrame,
    columns: list[str],
    categorical: set[str],
    correlation_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            left_categorical = left in categorical
            right_categorical = right in categorical
            if not left_categorical and not right_categorical:
                left_values = pd.to_numeric(frame[left], errors="coerce")
                right_values = pd.to_numeric(frame[right], errors="coerce")
                correlation = float(left_values.corr(right_values))
                if np.isfinite(correlation) and abs(correlation) >= correlation_threshold:
                    findings.append(
                        {
                            "left": left,
                            "right": right,
                            "reason": "high-numeric-correlation",
                            "absolute_correlation": abs(correlation),
                            "explanation": "两个数值变量提供高度重复的信息。",
                        }
                    )
                continue

            pair = frame[[left, right]].dropna().astype(str)
            left_to_right = int(pair.groupby(left, observed=True)[right].nunique().max()) <= 1
            right_to_left = int(pair.groupby(right, observed=True)[left].nunique().max()) <= 1
            if left_to_right and right_to_left:
                findings.append(
                    {
                        "left": left,
                        "right": right,
                        "reason": "deterministic-reencoding",
                        "explanation": "两个变量在当前数据中几乎是同一信息的不同编码。",
                    }
                )
    return findings


def atomic_write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"Output already exists: {path}")
    atomic_replace_text(path, content)


def atomic_write_json(path: Path, payload: Any, overwrite: bool) -> None:
    """Serialize JSON in the proposal's stable format and write it atomically."""
    atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        overwrite,
    )


def main() -> None:
    args = parse_args()
    if not 0.5 < args.confidence_level < 1:
        fail("confidence-level must be between 0.5 and 1")

    task_path = Path(args.task).resolve()
    workflow_path = Path(args.workflow_support).resolve()
    prep_path = Path(args.preparation_log).resolve()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    task = read_json(task_path, "analysis task")
    workflow_support = read_json(workflow_path, "workflow support assessment")
    prep = read_json(prep_path, "preparation log")
    if (
        workflow_support.get("status") != "supported"
        or workflow_support.get("execution_allowed") is not True
        or workflow_support.get("user_confirmed") is not True
        or not isinstance(workflow_support.get("approval"), dict)
    ):
        reasons = [
            str(item.get("required_workflow") or item.get("code"))
            for item in workflow_support.get("blocking_reasons", [])
            if isinstance(item, dict)
        ]
        detail = ", ".join(reasons) if reasons else "unresolved structure information"
        fail(
            "Current workflow is not approved for modeling: "
            f"{workflow_support.get('status')}; {detail}"
        )
    if workflow_support.get("analysis_task_sha256") != sha256_file(task_path):
        fail("Workflow support assessment does not match the current analysis task")
    if args.model_type not in workflow_support.get("supported_models", []):
        fail(f"Model type {args.model_type!r} is not allowed by the workflow assessment")
    if prep.get("data_preparation_executed") is not True:
        fail("Preparation log must contain data_preparation_executed: true")
    if prep.get("modeling_executed") is not False:
        fail("Preparation log must contain modeling_executed: false")
    if task.get("status") != "approved" or not isinstance(task.get("approval"), dict):
        fail("Model proposal requires approved-analysis-task.json")
    if not data_path.is_file():
        fail(f"Cleaned data does not exist: {data_path}")

    outcome = task.get("outcome")
    predictors = task.get("predictors", [])
    controls = task.get("controls", [])
    if not isinstance(outcome, str) or not outcome:
        fail("Analysis task must identify one outcome")
    if not isinstance(predictors, list) or not predictors or not all(isinstance(x, str) for x in predictors):
        fail("Analysis task must identify at least one predictor")
    if not isinstance(controls, list) or not all(isinstance(x, str) for x in controls):
        fail("Analysis task controls must be a list")
    selected = [outcome, *predictors, *controls]
    if len(set(selected)) != len(selected):
        fail("Outcome, predictors, and controls must not overlap")

    df = read_csv(data_path)
    missing_columns = [column for column in selected if column not in df.columns]
    if missing_columns:
        fail("Cleaned data is missing variables: " + ", ".join(missing_columns))
    exclusions = prep.get("analysis_metadata", {}).get("analysis_exclusions", [])
    blocked = [column for column in selected if column in exclusions]
    if blocked:
        fail("Approved task uses columns excluded during preparation: " + ", ".join(blocked))

    task_types = {
        item.get("column"): item.get("inferred_type")
        for item in task.get("variables", [])
        if isinstance(item, dict)
    }
    categorical = set(prep.get("analysis_metadata", {}).get("categorical_columns", []))
    for column in [*predictors, *controls]:
        inferred = task_types.get(column)
        if inferred in {"text", "categorical", "boolean"}:
            categorical.add(column)
    categorical &= set([*predictors, *controls])
    continuous_columns = sorted(
        column for column in [*predictors, *controls] if column not in categorical
    )
    requested_continuous_forms = parse_continuous_forms(args.continuous_form)
    unknown_form_columns = sorted(set(requested_continuous_forms) - set(continuous_columns))
    if unknown_form_columns:
        fail(
            "continuous-form names columns that are not continuous predictors/controls: "
            + ", ".join(unknown_form_columns)
        )
    if args.model_type not in SUPPORTED_MODEL_TYPE_SET and requested_continuous_forms:
        fail("continuous-form is unsupported for this model type")

    model_frame = df[selected].copy()
    complete = model_frame.dropna()
    excluded_rows = [int(index) + 2 for index in model_frame.index[model_frame.isna().any(axis=1)]]
    if complete.empty:
        fail("No complete rows remain for the approved variables")
    unique = sorted(complete[outcome].dropna().astype(str).unique().tolist())

    y_numeric: pd.Series | None = None
    if args.model_type in {"ols", "poisson", "negative-binomial"}:
        y_numeric = pd.to_numeric(complete[outcome], errors="coerce")
        if y_numeric.isna().any():
            fail(f"Outcome {outcome!r} must be numeric for the selected model")
    if args.model_type == "ols" and y_numeric is not None and y_numeric.nunique() < 2:
        fail("OLS outcome must vary")
    if args.model_type == "logistic":
        if len(unique) != 2:
            fail("Logistic regression requires exactly two outcome categories")
        if args.positive_class is None:
            fail("Logistic regression requires --positive-class after user confirmation")
        if str(args.positive_class) not in unique:
            fail(f"positive-class must be one of {unique}")
    elif args.positive_class is not None:
        fail("positive-class is only valid for logistic regression")
    if args.model_type in {"poisson", "negative-binomial"}:
        assert y_numeric is not None
        values = y_numeric.to_numpy(dtype=float)
        if (values < 0).any() or not np.allclose(values, np.round(values)):
            fail("Count-regression outcomes must contain non-negative integers")
        if args.model_type == "negative-binomial" and float(np.var(values)) <= float(np.mean(values)):
            warnings_for_outcome = [
                "Outcome variance does not exceed its mean; confirm why negative binomial is preferred over Poisson."
            ]
        else:
            warnings_for_outcome = []
    else:
        warnings_for_outcome = []

    outcome_categories: list[str] | None = None
    reference_class: str | None = None
    if args.model_type == "multinomial-logistic":
        if len(unique) < 3:
            fail("Multinomial logistic regression requires at least three unordered outcome categories")
        if args.reference_class is None:
            fail("Multinomial logistic regression requires --reference-class after user confirmation")
        reference_class = str(args.reference_class)
        if reference_class not in unique:
            fail(f"reference-class must be one of {unique}")
        outcome_categories = [reference_class, *(value for value in unique if value != reference_class)]
    elif args.reference_class is not None:
        fail("reference-class is only valid for multinomial logistic regression")

    if args.model_type == "ordinal-logistic":
        if len(unique) < 3:
            fail("Ordinal logistic regression requires at least three ordered outcome categories")
        if not args.category_order:
            fail("Ordinal logistic regression requires --category-order after user confirmation")
        supplied_order = [str(value) for value in args.category_order]
        if len(supplied_order) != len(set(supplied_order)):
            fail("category-order must not contain duplicate categories")
        if set(supplied_order) != set(unique):
            fail(f"category-order must contain each observed category exactly once: {unique}")
        outcome_categories = supplied_order
    elif args.category_order is not None:
        fail("category-order is only valid for ordinal logistic regression")

    requested_references = parse_categorical_references(args.categorical_reference)
    unknown_reference_columns = sorted(set(requested_references) - categorical)
    if unknown_reference_columns:
        fail(
            "categorical-reference names columns that are not categorical predictors/controls: "
            + ", ".join(unknown_reference_columns)
        )
    metadata = task_variable_metadata(task)
    categorical_references: dict[str, str] = {}
    categorical_reference_sources: dict[str, str] = {}
    for column in sorted(categorical):
        values = complete[column].astype(str)
        observed = sorted(values.unique().tolist())
        metadata_reference = metadata.get(column, {}).get("reference_category")
        if column in requested_references:
            reference = requested_references[column]
            source = "explicit-command"
        elif isinstance(metadata_reference, str) and metadata_reference.strip():
            reference = metadata_reference.strip()
            source = "approved-task-metadata"
        else:
            counts = values.value_counts()
            largest = int(counts.max())
            reference = sorted(str(value) for value, count in counts.items() if int(count) == largest)[0]
            source = "automatic-most-frequent"
        if reference not in observed:
            fail(
                f"Reference category {reference!r} is not observed for {column!r}: {observed}"
            )
        categorical_references[column] = reference
        categorical_reference_sources[column] = source

    collinearity_candidates = screen_collinearity_candidates(
        complete,
        [*predictors, *controls],
        categorical,
    )
    blocking_redundancies = [
        item
        for item in collinearity_candidates
        if item.get("reason") == "deterministic-reencoding"
    ]

    x = complete[[*predictors, *controls]].copy()
    for column in x.columns:
        if column not in categorical:
            x[column] = pd.to_numeric(x[column], errors="coerce")
            if x[column].isna().any():
                fail(f"Non-categorical predictor {column!r} contains non-numeric values")
    resolved_continuous_forms, continuous_form_decisions = resolve_continuous_forms(
        complete,
        continuous_columns,
        requested_continuous_forms,
        metadata,
        args.model_type,
    )
    continuous_form_specs = continuous_form_specifications(
        complete,
        continuous_columns,
        resolved_continuous_forms,
        outcome,
        (str(args.positive_class) if args.positive_class is not None else None),
        args.model_type,
        outcome_categories,
    )
    for column, decision in continuous_form_decisions.items():
        continuous_form_specs[column]["selection_source"] = decision["source"]
        continuous_form_specs[column]["selection_rationale"] = decision["rationale"]
    x = apply_continuous_forms(x, continuous_form_specs)
    x = encode_categorical_predictors(x, categorical, categorical_references)
    design_preflight = design_matrix_preflight(x, args.model_type)
    if design_preflight["model_fitting_allowed"] is not True:
        preflight_path = output_dir / "design-matrix-preflight.json"
        atomic_write(
            preflight_path,
            json.dumps(design_preflight, ensure_ascii=False, indent=2) + "\n",
            args.overwrite,
        )
        fail(
            "Final design-matrix preflight failed before model approval; "
            f"rank={design_preflight.get('rank')}, columns={design_preflight.get('column_count')}, "
            f"dependent_columns={design_preflight.get('dependent_columns')}. "
            f"Review {preflight_path} and revise the data structure before regenerating the proposal."
        )
    base_parameter_count = 1 + len(x.columns)
    if args.model_type == "multinomial-logistic":
        assert outcome_categories is not None
        parameter_count = base_parameter_count * (len(outcome_categories) - 1)
    elif args.model_type == "ordinal-logistic":
        assert outcome_categories is not None
        parameter_count = len(x.columns) + len(outcome_categories) - 1
    else:
        parameter_count = base_parameter_count
    category_support_screen = screen_category_support(
        complete,
        outcome=outcome,
        categorical_predictors=categorical,
        model_type=args.model_type,
        parameters_per_outcome_equation=base_parameter_count,
        design_matrix=x,
        positive_class=(str(args.positive_class) if args.model_type == "logistic" else None),
    )
    iia_check = iia_not_applicable()
    if args.model_type == "multinomial-logistic":
        assert outcome_categories is not None
        if category_support_screen.get("model_fitting_allowed") is False or blocking_redundancies:
            iia_check = {
                "status": "not-evaluated",
                "model_fitting_allowed": False,
                "reason": "category-support or deterministic-collinearity blockers must be resolved first",
                "plain_explanation": "IIA检查尚未执行，因为当前规格已有更早的模型识别阻断项。",
                "required_action": "resolve earlier blockers and regenerate the proposal",
            }
        else:
            try:
                iia_check = screen_iia(
                    complete,
                    outcome,
                    sm.add_constant(x, has_constant="add"),
                    outcome_categories,
                    1.0 - args.confidence_level,
                )
            except Exception as exc:
                iia_check = {
                    "status": "not-evaluated",
                    "model_fitting_allowed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "plain_explanation": "IIA检查未能可靠完成；这不能解释为IIA假设成立。",
                    "required_action": "use a specialized multinomial choice workflow or revise the outcome structure",
                }
    proportional_odds_check = not_applicable()
    if args.model_type == "ordinal-logistic":
        assert outcome_categories is not None
        if category_support_screen.get("model_fitting_allowed") is False or blocking_redundancies:
            proportional_odds_check = {
                "status": "not-evaluated",
                "model_fitting_allowed": False,
                "reason": "category-support or deterministic-collinearity blockers must be resolved first",
                "plain_explanation": "比例优势检查尚未执行，因为当前规格已有更早的模型识别阻断项。",
                "required_action": "resolve earlier blockers and regenerate the proposal",
            }
        else:
            try:
                proportional_odds_check = screen_proportional_odds(
                    complete,
                    outcome,
                    x,
                    outcome_categories,
                    1.0 - args.confidence_level,
                )
            except Exception as exc:
                proportional_odds_check = {
                    "status": "not-evaluated",
                    "model_fitting_allowed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "plain_explanation": "比例优势检查未能可靠完成；这不能解释为假设成立。",
                    "required_action": "use multinomial logistic or a specialized ordinal workflow",
                }
    count_dispersion_check = count_dispersion_not_applicable()
    if args.model_type == "poisson":
        if blocking_redundancies:
            count_dispersion_check = {
                "status": "not-evaluated",
                "model_fitting_allowed": False,
                "reason": "deterministic-collinearity blockers must be resolved first",
                "plain_explanation": "Poisson过度离散检查尚未执行，因为当前变量组合不可识别。",
                "required_action": "resolve earlier blockers and regenerate the proposal",
            }
        else:
            try:
                count_dispersion_check = screen_count_dispersion(
                    complete,
                    outcome,
                    sm.add_constant(x, has_constant="add"),
                    1.0 - args.confidence_level,
                )
            except Exception as exc:
                count_dispersion_check = {
                    "status": "not-evaluated",
                    "model_fitting_allowed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "plain_explanation": "Poisson过度离散检查未能可靠完成；这不能解释为Poisson假设成立。",
                    "required_action": "return to model choice or stop for an explanation",
                }
    negative_binomial_need_check = negative_binomial_need_not_applicable()
    if args.model_type == "negative-binomial":
        if blocking_redundancies:
            negative_binomial_need_check = {
                "status": "not-evaluated",
                "model_fitting_allowed": False,
                "reason": "deterministic-collinearity blockers must be resolved first",
                "plain_explanation": "负二项必要性检查尚未执行，因为当前变量组合不可识别。",
                "required_action": "resolve earlier blockers and regenerate the proposal",
            }
        else:
            try:
                negative_binomial_need_check = screen_negative_binomial_need(
                    complete,
                    outcome,
                    sm.add_constant(x, has_constant="add"),
                    1.0 - args.confidence_level,
                )
            except Exception as exc:
                negative_binomial_need_check = {
                    "status": "not-evaluated",
                    "model_fitting_allowed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "plain_explanation": "负二项必要性检查未能可靠完成；这不能解释为负二项模型有必要。",
                    "required_action": "return to model choice or stop for an explanation",
                }
    zero_inflation_check = zero_inflation_not_applicable()
    if args.model_type in {"poisson", "negative-binomial"}:
        earlier_count_gate_allowed = (
            count_dispersion_check.get("model_fitting_allowed") is True
            if args.model_type == "poisson"
            else negative_binomial_need_check.get("model_fitting_allowed") is True
        )
        if blocking_redundancies or not earlier_count_gate_allowed:
            zero_inflation_check = {
                "status": "not-evaluated",
                "model_fitting_allowed": False,
                "reason": "an earlier count-model identification or dispersion gate must be resolved first",
                "plain_explanation": "过多零值检查尚未执行，因为当前计数模型已有更早的阻断项。",
                "required_action": "resolve the earlier count-model gate and regenerate the proposal",
            }
        else:
            try:
                zero_inflation_check = screen_zero_inflation(
                    complete,
                    outcome,
                    sm.add_constant(x, has_constant="add"),
                    args.model_type,
                    1.0 - args.confidence_level,
                )
            except Exception as exc:
                zero_inflation_check = {
                    "status": "not-evaluated",
                    "model_fitting_allowed": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "plain_explanation": "过多零值检查未能可靠完成；这不能解释为零值结构合适。",
                    "required_action": "use a specialized hurdle/zero-inflated workflow or stop",
                }
    n = len(complete)
    residual_df = n - parameter_count
    if residual_df <= 0:
        fail(
            f"Model is not identifiable: {n} complete rows for approximately {parameter_count} parameters"
        )
    warnings: list[str] = [*warnings_for_outcome]
    if collinearity_candidates:
        pairs = ", ".join(
            f"{item['left']}↔{item['right']}" for item in collinearity_candidates
        )
        warnings.append(
            "Potentially redundant predictor encodings require user review before "
            f"individual-effect interpretation: {pairs}."
        )
    if residual_df < 10:
        warnings.append("Residual degrees of freedom are below 10; estimates and diagnostics will be unstable.")
    if n < 30:
        warnings.append("Sample size is below 30; treat inference as exploratory.")
    if category_support_screen.get("status") in {"requires-review", "revision-required"}:
        warnings.append(
            "Categorical support or separation risk requires a user-approved revision before fitting; "
            "see category-support-screen.csv."
        )
    if proportional_odds_check.get("status") in {"violation-detected", "not-evaluated"}:
        warnings.append(
            "Ordinary ordinal logistic is blocked because the proportional-odds assumption was violated or could not be evaluated; "
            "use multinomial logistic or a specialized partial-proportional-odds workflow."
        )
    if iia_check.get("status") in {"sensitivity-detected", "not-evaluated"}:
        warnings.append(
            "Ordinary multinomial logistic is blocked because IIA sensitivity was detected or could not be evaluated reliably; "
            "use a specialized nested/multinomial-probit choice workflow or revise the outcome structure."
        )
    if count_dispersion_check.get("status") in {"overdispersion-detected", "not-evaluated"}:
        warnings.append(
            "Ordinary Poisson is blocked because adjusted overdispersion was detected or could not be evaluated reliably."
        )
    if negative_binomial_need_check.get("status") in {
        "no-detected-need-for-extra-dispersion", "not-evaluated"
    }:
        warnings.append(
            "Negative-binomial regression is blocked because adjusted extra dispersion was not detected or could not be evaluated."
        )
    if zero_inflation_check.get("status") in {"excess-zeros-detected", "not-evaluated"}:
        warnings.append(
            "Ordinary count regression is blocked because excess zeros were detected or could not be evaluated after the earlier count-model gates."
        )
    if task.get("goal") == "causal":
        fail("Causal modeling requires a dedicated approved design workflow")

    proposed_at = datetime.now().astimezone().isoformat()
    spec = {
        "status": "draft",
        "requires_user_confirmation": True,
        "user_confirmed": False,
        "modeling_executed": False,
        "proposed_at": proposed_at,
        "title": task.get("title", "Statistical analysis"),
        "goal": task.get("goal", "association"),
        "unit_of_analysis": task.get("unit_of_analysis"),
        "outcome": outcome,
        "predictors": predictors,
        "controls": controls,
        "categorical_columns": sorted(categorical),
        "model_type": args.model_type,
        "robust_se": args.robust_se,
        "confidence_level": args.confidence_level,
        "positive_class": args.positive_class,
        "outcome_reference_class": (
            next(value for value in unique if value != str(args.positive_class))
            if args.model_type == "logistic"
            else None
        ),
        "reference_class": reference_class,
        "outcome_categories": outcome_categories,
        "categorical_reference_categories": categorical_references,
        "categorical_reference_sources": categorical_reference_sources,
        "continuous_functional_forms": continuous_form_specs,
        "design_matrix_preflight": design_preflight,
        "category_support_screen": category_support_screen,
        "iia_check": iia_check,
        "proportional_odds_check": proportional_odds_check,
        "count_dispersion_check": count_dispersion_check,
        "negative_binomial_need_check": negative_binomial_need_check,
        "zero_inflation_check": zero_inflation_check,
        "collinearity_screen": {
            "status": (
                "revision-required"
                if blocking_redundancies
                else ("requires-review" if collinearity_candidates else "clear")
            ),
            "model_fitting_allowed": not blocking_redundancies,
            "numeric_correlation_threshold": 0.95,
            "candidate_pairs": collinearity_candidates,
            "blocking_pairs": blocking_redundancies,
            "plain_explanation": (
                "共线性表示多个自变量包含大量重复信息；VIF越高，越难稳定拆分各变量的独立作用。"
            ),
        },
        "complete_case_rows": n,
        "excluded_cleaned_data_rows": excluded_rows,
        "estimated_parameter_count": parameter_count,
        "estimated_residual_df": residual_df,
        "warnings": warnings,
        "workflow_support": {
            "status": workflow_support.get("status"),
            "execution_allowed": workflow_support.get("execution_allowed"),
            "checked_dimensions": workflow_support.get("checked_dimensions"),
            "assessed_at": workflow_support.get("assessed_at"),
        },
        "provenance": {
            "analysis_task": str(task_path),
            "analysis_task_sha256": sha256_file(task_path),
            "workflow_support_assessment": str(workflow_path),
            "workflow_support_assessment_sha256": sha256_file(workflow_path),
            "preparation_log": str(prep_path),
            "preparation_log_sha256": sha256_file(prep_path),
            "cleaned_data": str(data_path),
            "cleaned_data_sha256": sha256_file(data_path),
        },
    }
    md = "\n".join(
        [
            "# 待批准的模型规格提案",
            "",
            f"- 模型：`{args.model_type}`",
            f"- 因变量：`{outcome}`",
            f"- 自变量：{', '.join(f'`{x}`' for x in predictors)}",
            f"- 控制变量：{', '.join(f'`{x}`' for x in controls) if controls else '无'}",
            f"- 分类变量：{', '.join(f'`{x}`' for x in sorted(categorical)) if categorical else '无'}",
            f"- 最终设计矩阵预检：`{design_preflight['status']}`；{design_preflight['row_count']} 行、{design_preflight['column_count']} 列、秩 {design_preflight['rank']}",
            *(
                [f"- 二分类目标事件：`{args.positive_class}`；结果基准类别：`{next(value for value in unique if value != str(args.positive_class))}`"]
                if args.model_type == "logistic"
                else []
            ),
            *(
                ["- 分类自变量参照组："]
                + [
                    f"  - `{column}`：`{categorical_references[column]}`（"
                    + {
                        "explicit-command": "本次明确指定",
                        "approved-task-metadata": "已批准任务中指定",
                        "automatic-most-frequent": "自动提议：样本量最多类别",
                    }.get(categorical_reference_sources[column], "来源待核对")
                    + "）"
                    for column in sorted(categorical_references)
                ]
                if categorical_references
                else []
            ),
            *(
                [
                    "- 共线性预检查：发现可能重复表达同一信息的变量组合。",
                    "  - 简明说明：共线性表示多个自变量含有大量重复信息；VIF越高，越难稳定拆分各变量的独立作用。",
                    *(
                        f"  - `{item['left']}` ↔ `{item['right']}`：{item['explanation']}"
                        for item in collinearity_candidates
                    ),
                    *(
                        ["  - 其中存在确定性重复编码，当前规格不可拟合；请修改变量组合后重新生成提案。"]
                        if blocking_redundancies
                        else ["  - 请在批准时选择修改变量组合，或保留模型但接受相关单项系数不作稳定解释。"]
                    ),
                ]
                if collinearity_candidates
                else ["- 共线性预检查：未发现高度相关或确定性重复编码的变量组合。"]
            ),
            *(
                ["- 连续变量函数形式（已纳入本次批准范围）："]
                + [
                    f"  - `{column}`：{continuous_form_specs[column]['plain_label']}"
                    + f"（{CONTINUOUS_FORM_SOURCE_LABELS.get(continuous_form_specs[column]['selection_source'], continuous_form_specs[column]['selection_source'])}：{continuous_form_specs[column]['selection_rationale']}）"
                    + (
                        "；结点（原单位）="
                        + "、".join(f"{value:.4g}" for value in continuous_form_specs[column]["knots_original_units"])
                        if continuous_form_specs[column]["form"] == "restricted-cubic-spline"
                        else ""
                    )
                    + "；"
                    + format_continuous_preview(continuous_form_specs[column])
                    for column in continuous_columns
                ]
                if continuous_form_specs
                else []
            ),
            *(
                [
                    f"- 类别支持度检查：`{category_support_screen.get('status')}`",
                    "  - 简明说明：稀疏类别是某个类别样本或某种结果太少；分类分离是某类别几乎或完全只对应一种结果，会让普通Logistic系数和优势比失控。",
                    f"  - 分离阻断项：{len(category_support_screen.get('blocking_findings', []))}；其他稀疏或模型信息量警告：{len(category_support_screen.get('review_findings', []))}",
                    "  - 完整的每个类别×结果计数见 `category-support-screen.csv`。系统不会自动合并或删除类别。",
                    *(
                        ["  - 当前规格不可拟合；请选择按业务含义合并类别、在新版任务中排除相关因素、补充数据或停止分析。"]
                        if category_support_screen.get("model_fitting_allowed") is False
                        else []
                    ),
                ]
                if category_support_screen.get("status") != "not-applicable"
                else []
            ),
            f"- 完整案例数：{n}",
            f"- 估计参数数：{parameter_count}",
            f"- 置信水平：{args.confidence_level:.1%}",
            f"- 稳健标准误：`{args.robust_se}`",
            "- 工作流适配性：`supported`",
            *(
                [f"- 无序分类参照类别：`{reference_class}`"]
                if reference_class is not None
                else []
            ),
            *(
                [f"- 结果类别顺序：{' < '.join(f'`{value}`' for value in outcome_categories)}"]
                if args.model_type == "ordinal-logistic" and outcome_categories
                else []
            ),
            *(
                [
                    f"- IIA假设检查：`{iia_check.get('status')}`",
                    f"  - {iia_check.get('plain_explanation')}",
                    f"  - 最小Holm校正p值：{iia_check.get('minimum_adjusted_p_value', '未得到')}；已批准判断水平：{1.0 - args.confidence_level:.3g}",
                    f"  - 后续要求：{iia_check.get('required_action')}",
                ]
                if args.model_type == "multinomial-logistic"
                else []
            ),
            *(
                [
                    f"- 比例优势假设检查：`{proportional_odds_check.get('status')}`",
                    f"  - {proportional_odds_check.get('plain_explanation')}",
                    f"  - 检验p值：{proportional_odds_check.get('p_value', '未得到')}；已批准判断水平：{1.0 - args.confidence_level:.3g}",
                    f"  - 后续要求：{proportional_odds_check.get('required_action')}",
                ]
                if args.model_type == "ordinal-logistic"
                else []
            ),
            *(
                [
                    f"- Poisson过度离散检查：`{count_dispersion_check.get('status')}`",
                    f"  - {count_dispersion_check.get('plain_explanation')}",
                    f"  - 单侧检验p值：{count_dispersion_check.get('p_value', '未得到')}；Pearson离散度（仅描述）：{count_dispersion_check.get('pearson_dispersion_descriptive', '未得到')}",
                    f"  - 后续要求：{count_dispersion_check.get('required_action')}",
                ]
                if args.model_type == "poisson"
                else []
            ),
            *(
                [
                    f"- 负二项必要性检查：`{negative_binomial_need_check.get('status')}`",
                    f"  - {negative_binomial_need_check.get('plain_explanation')}",
                    f"  - 单侧检验p值：{negative_binomial_need_check.get('p_value', '未得到')}；后续要求：{negative_binomial_need_check.get('required_action')}",
                ]
                if args.model_type == "negative-binomial"
                else []
            ),
            *(
                [
                    f"- 过多零值检查：`{zero_inflation_check.get('status')}`",
                    f"  - {zero_inflation_check.get('plain_explanation')}",
                    f"  - 实际零值={zero_inflation_check.get('observed_zero_count', '未得到')}；模型预计零值={zero_inflation_check.get('expected_zero_count', '未得到')}；单侧p值={zero_inflation_check.get('p_value', '未得到')}",
                ]
                if args.model_type in {"poisson", "negative-binomial"}
                else []
            ),
            "",
            "## 警告",
            "",
            *(f"- {warning}" for warning in warnings),
            "- 该规格不会把统计关联解释为因果关系。",
            "",
        ]
    )
    json_path = output_dir / "model-specification-proposal.json"
    md_path = output_dir / "model-specification-proposal.md"
    support_path = output_dir / "category-support-screen.csv"
    iia_path = output_dir / "iia-check.json"
    proportional_odds_path = output_dir / "proportional-odds-check.json"
    count_dispersion_path = output_dir / "count-dispersion-check.json"
    negative_binomial_need_path = output_dir / "negative-binomial-need-check.json"
    zero_inflation_path = output_dir / "zero-inflation-check.json"
    atomic_write_json(json_path, spec, args.overwrite)
    atomic_write(md_path, md, args.overwrite)
    support_rows = flatten_category_support_rows(category_support_screen)
    support_columns = [
        "factor",
        "level",
        "total",
        *(
            f"outcome_count:{category}"
            for category in category_support_screen.get("outcome_categories", [])
        ),
        "minimum_observed_outcome_count",
        "minimum_expected_outcome_count",
        "zero_outcome_categories",
        "risk_codes",
        "blocking",
    ]
    atomic_write(
        support_path,
        pd.DataFrame(support_rows, columns=support_columns).to_csv(index=False),
        args.overwrite,
    )
    atomic_write_json(iia_path, iia_check, args.overwrite)
    atomic_write_json(proportional_odds_path, proportional_odds_check, args.overwrite)
    atomic_write_json(count_dispersion_path, count_dispersion_check, args.overwrite)
    atomic_write_json(
        negative_binomial_need_path, negative_binomial_need_check, args.overwrite
    )
    atomic_write_json(zero_inflation_path, zero_inflation_check, args.overwrite)
    print(json.dumps({"ok": True, "status": "draft", "requires_user_confirmation": True, "model_type": args.model_type, "complete_rows": n, "warnings": len(warnings), "outputs": [str(md_path), str(json_path), str(support_path), str(iia_path), str(proportional_odds_path), str(count_dispersion_path), str(negative_binomial_need_path), str(zero_inflation_path)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
