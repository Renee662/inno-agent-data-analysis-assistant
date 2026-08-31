#!/usr/bin/env python3
"""Generate a polished offline final-report.html from approved analysis artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from approval import ApprovalError, approval_summary, verify_approval  # noqa: E402
from category_support import flatten_category_support_rows  # noqa: E402
from file_utils import sha256_file  # noqa: E402
from model_registry import MODEL_LABELS, SUPPORTED_MODEL_TYPE_SET  # noqa: E402


CONTINUOUS_FORM_SOURCE_LABELS = {
    "explicit-approved-domain-override": "已批准的专业设定",
    "approved-task-metadata": "已批准任务中的专业设定",
    "automatic-fixed-flexible-default": "系统固定的灵活默认规则",
    "automatic-limited-support-linear": "取值支持不足时的保守直线规则",
}

SENSITIVE_SEMANTIC_RE = re.compile(
    r"(?:^|[_\-\s])(sex|gender|race|ethnicity|marital.?status|nationality|native.?country)(?:$|[_\-\s])|"
    r"性别|种族|族裔|民族|婚姻|国籍|原籍国",
    re.IGNORECASE,
)

GOAL_LABELS = {
    "association": "关联分析",
    "prediction": "预测分析",
    "description": "描述性分析",
    "causal": "因果分析",
}

DECISION_GOAL_LABELS = {
    "relationships": "了解几个因素分别与结果有什么关系",
    "prediction": "尽可能准确地预测结果",
    "group-comparison": "比较不同群体的平均差异",
}

CHOICE_LABELS = {
    "keep": "保留",
    "drop-exact-duplicates": "删除完全重复记录",
    "deduplicate-with-key": "按关键字段去重",
    "complete-case": "完整案例处理",
    "median-imputation": "中位数填补",
    "missing-indicator": "增加缺失指示变量",
    "model-specific-method": "延后到模型特定方法",
    "keep-and-run-diagnostics": "保留并进行影响诊断",
    "transform": "变量变换",
    "winsorize": "缩尾处理",
    "exclude-with-domain-rule": "按领域规则排除",
    "treat-as-categorical": "作为分类变量",
    "keep-as-text-and-exclude": "保留文本但不纳入模型",
    "manual-recode": "人工重编码",
}

METRIC_LABELS = {
    "n": "模型样本数",
    "parameters": "估计参数数",
    "df_resid": "残差自由度",
    "aic": "AIC",
    "bic": "BIC",
    "r_squared": "R²",
    "adjusted_r_squared": "调整 R²",
    "rmse": "均方根误差（RMSE）",
    "durbin_watson": "Durbin–Watson",
    "f_p_value": "整体 F 检验 p 值",
    "breusch_pagan_p_value": "Breusch–Pagan p 值",
    "jarque_bera_p_value": "Jarque–Bera p 值",
    "log_likelihood": "对数似然",
    "mcfadden_pseudo_r_squared": "McFadden 伪 R²",
    "accuracy_at_0_5": "0.5 阈值准确率",
    "roc_auc": "ROC AUC",
    "brier_score": "Brier 分数",
    "calibration_mean_absolute_error": "分组校准平均绝对误差",
    "sensitivity_at_0_5": "0.5阈值灵敏度",
    "specificity_at_0_5": "0.5阈值特异度",
    "deviance": "偏差（Deviance）",
    "pearson_chi2": "Pearson χ²",
    "dispersion": "离散度",
    "llr_p_value": "整体似然比检验 p 值",
    "negative_binomial_alpha": "负二项离散参数 α",
    "classification_accuracy": "样本内分类准确率",
    "outcome_categories": "结果类别数",
    "multiclass_log_loss": "多分类对数损失",
    "ordinal_mean_absolute_category_error": "有序类别平均绝对误差",
    "count_rmse": "计数 RMSE",
    "cross_validated_accuracy": "折外准确率",
    "cross_validated_accuracy_at_0_5": "折外0.5阈值准确率",
    "cross_validated_log_loss": "折外对数损失",
    "cross_validated_brier_score": "折外Brier分数",
    "cross_validated_roc_auc": "折外ROC AUC",
    "cross_validated_rmse": "折外RMSE",
    "cross_validated_mae": "折外MAE",
    "cross_validated_r_squared": "折外R²",
    "cross_validated_ordinal_mean_absolute_category_error": "折外平均绝对等级误差",
}

METRIC_DESCRIPTIONS = {
    "n": "实际进入模型计算的记录数。",
    "parameters": "模型同时估计的系数个数，包含截距和分类变量展开后的系数。",
    "df_resid": "扣除已估计参数后，仍可用于估计随机误差的信息量。",
    "aic": "用于比较同一数据、同一结果变量上的候选模型；数值本身不能单独判断模型好坏，通常越小越有利。",
    "bic": "用于比较同一数据上的候选模型，并比 AIC 更强地惩罚复杂模型；数值本身不能单独解释。",
    "r_squared": "当前样本中，模型所解释的结果变量差异比例。",
    "adjusted_r_squared": "在 R² 基础上考虑模型纳入因素数量，便于识别因增加变量造成的表面提升。",
    "rmse": "模型拟合值与实际值之间的典型偏差，单位与结果变量相同。",
    "durbin_watson": "检查相邻残差是否同向聚集；接近 2 通常表示未见明显的一阶相关。",
    "f_p_value": "检验所有非截距系数是否同时为 0。",
    "breusch_pagan_p_value": "检查残差波动幅度是否随拟合水平系统变化。",
    "jarque_bera_p_value": "检查残差分布是否明显偏离正态形态。",
    "log_likelihood": "衡量模型对观测数据的相对拟合程度，主要用于候选模型比较。",
    "mcfadden_pseudo_r_squared": "Logistic 模型相对仅含截距模型的拟合改善程度，不等同于线性模型的 R²。",
    "accuracy_at_0_5": "以 0.5 为分类阈值时预测正确的比例，需结合类别比例理解。",
    "roc_auc": "模型区分正类与负类的能力；0.5 接近随机，1 表示完全区分。",
    "brier_score": "预测概率与实际二分类结果之间的均方误差，越小越好。",
    "calibration_mean_absolute_error": "各预测概率组的平均预测概率与实际正类比例之间的平均绝对差。",
    "sensitivity_at_0_5": "以0.5为阈值时正确识别正类的比例。",
    "specificity_at_0_5": "以0.5为阈值时正确识别负类的比例。",
    "deviance": "计数模型未解释偏差的汇总量，主要用于模型比较和拟合检查。",
    "pearson_chi2": "计数模型残差差异的汇总量，常用于计算离散度。",
    "dispersion": "计数数据实际波动与 Poisson 模型预期波动之比。",
    "llr_p_value": "比较当前模型与仅含截距基准模型的整体似然比检验。",
    "negative_binomial_alpha": "负二项模型用于容纳额外计数波动的离散参数；大于 0 表示方差可超过均值。",
    "classification_accuracy": "按最大预测概率选择类别时，在当前样本中分类正确的比例；不是新数据准确率。",
    "outcome_categories": "当前结果变量纳入模型的类别总数。",
    "multiclass_log_loss": "衡量模型为实际类别分配概率的质量，错误且自信的预测惩罚更大。",
    "ordinal_mean_absolute_category_error": "预测等级与实际等级平均相差多少级。",
    "count_rmse": "拟合计数与实际计数之间的典型差距。",
}

FIGURE_METADATA = {
    "eda-distributions.png": (
        "主要变量的分布",
        "展示数值变量的分布形态和分类变量的频数，用于识别偏态、极端值和类别不平衡。",
    ),
    "eda-relationships.png": (
        "主要变量之间的关系",
        "展示关注因素与结果指标之间的原始数据关系；图中趋势尚未控制其他因素。",
    ),
    "eda-correlations.png": (
        "数值变量相关矩阵",
        "相关系数描述两两线性关系，不能单独解释为因果关系。",
    ),
    "coefficients.png": (
        "模型系数与置信区间",
        "点表示系数估计值，横线表示置信区间；零线用于判断区间是否覆盖无效应值。",
    ),
    "residuals-vs-fitted.png": (
        "残差与拟合值",
        "用于检查非线性、异方差和可能遗漏的结构；理想情况下点应围绕零线随机分布。",
    ),
    "residual-qq.png": (
        "残差 Q-Q 图",
        "用于比较残差分位数与正态理论分位数；系统性偏离参考线提示分布假设可能不充分。",
    ),
    "logistic-roc.png": ("Logistic ROC 曲线", "展示不同阈值下真阳性率与假阳性率的权衡，只反映区分能力。"),
    "logistic-calibration.png": ("Logistic 校准图", "比较平均预测概率与实际正类比例；越接近对角线，概率解释越可靠。"),
    "count-residuals-vs-fitted.png": ("计数模型残差与拟合值", "检查计数残差是否围绕零线随机分布以及是否存在系统结构。"),
    "count-observed-vs-fitted.png": ("实际计数与拟合计数", "比较实际计数和模型拟合计数；对角线表示完全一致。"),
    "classification-confusion.png": ("分类混淆矩阵", "按实际类别和预测类别汇总记录数，用于定位集中错分。"),
    "classification-calibration.png": ("分类概率校准", "多分类模型逐类别比较分组预测概率与实际发生率；有序模型逐等级分界比较累计预测概率与累计实际发生率。对角线表示理想校准。"),
    "influence.png": (
        "高影响观测诊断",
        "该图使用当前模型适用的杠杆值、残差及可用的Cook距离组合筛查候选记录；参考线只用于复核，不授权删除。",
    ),
    "relationship-preview.png": (
        "结果揭示前的观察图",
        "这张图用于“观察—判断—证据反馈”环节，展示尚未控制其他因素的原始关系。",
    ),
}

MAIN_DETAIL_LIMIT = 8
MAIN_SUMMARY_LIMIT = 3


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate outputs/final-report.html")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--analysis-task", required=True)
    parser.add_argument("--preparation-log", required=True)
    parser.add_argument("--missingness-impact", required=True)
    parser.add_argument("--preparation-plan")
    parser.add_argument("--model-spec", required=True)
    parser.add_argument("--model-results", required=True)
    parser.add_argument("--model-diagnostics", required=True)
    parser.add_argument("--factor-tests", required=True)
    parser.add_argument("--category-support", required=True)
    parser.add_argument("--shape-tests", required=True)
    parser.add_argument("--influence-diagnostics", required=True)
    parser.add_argument("--iia-check", required=True)
    parser.add_argument("--proportional-odds-check", required=True)
    parser.add_argument("--count-dispersion-check", required=True)
    parser.add_argument("--negative-binomial-need-check", required=True)
    parser.add_argument("--zero-inflation-check", required=True)
    parser.add_argument("--predictive-validation", required=True)
    parser.add_argument("--model-summary", required=True)
    parser.add_argument("--analysis-run-log", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--cleaned-data", required=True)
    parser.add_argument("--analysis-code", required=True)
    parser.add_argument("--approval-record", required=True)
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


def read_csv_rows(path: Path, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"{label} does not exist: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    fail(f"Cannot decode {label}: {last_error}")
    raise AssertionError("unreachable")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_number(value: Any, digits: int = 3) -> str:
    number = to_float(value)
    if number is None:
        return "—"
    absolute = abs(number)
    if absolute != 0 and (absolute < 0.001 or absolute >= 10000):
        return f"{number:.2e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def fmt_p(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "—"
    if number < 0.001:
        return "< 0.001"
    return f"{number:.3f}"


def fmt_p_statement(value: Any) -> str:
    """Format a p value as a complete statistical statement."""
    formatted = fmt_p(value)
    if formatted == "—":
        return "p 值不可用"
    return f"p {formatted}" if formatted.startswith("<") else f"p = {formatted}"


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def multiplicity_supported(row: dict[str, Any]) -> bool:
    return to_bool(row.get("multiplicity_supported")) is True


def interpretation_supported(row: dict[str, Any]) -> bool:
    recorded = to_bool(row.get("interpretation_supported"))
    return recorded is True if recorded is not None else multiplicity_supported(row)


def fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / 1024 / 1024:.1f} MiB"


def html_list(items: Iterable[str], class_name: str = "info-list") -> str:
    values = list(items)
    if not values:
        return '<p class="lead">无。</p>'
    return f'<ul class="{class_name}">' + "".join(f"<li>{item}</li>" for item in values) + "</ul>"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def copy_artifact(source: Path, destination: Path, overwrite: bool) -> None:
    if not source.is_file():
        fail(f"Required artifact does not exist: {source}")
    if destination.exists() and not overwrite:
        fail(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_artifacts(
    profile: dict[str, Any],
    task: dict[str, Any],
    prep: dict[str, Any],
    spec: dict[str, Any],
    summary: dict[str, Any],
    run_log: dict[str, Any],
    cleaned_data: Path,
    prep_path: Path,
    spec_path: Path,
    results: list[dict[str, str]],
    factor_tests: list[dict[str, str]],
    shape_tests: list[dict[str, str]],
    category_support_rows: list[dict[str, str]],
    influence_rows: list[dict[str, str]],
    iia: dict[str, Any],
    proportional_odds: dict[str, Any],
    count_dispersion: dict[str, Any],
    negative_binomial_need: dict[str, Any],
    zero_inflation: dict[str, Any],
    predictive_validation: dict[str, Any],
) -> None:
    if not profile.get("profiles"):
        fail("Data profile has no profiled sheet")
    if not isinstance(task.get("outcome"), str):
        fail("Analysis task has no outcome")
    for field, label in (("dataset_summary", "dataset summary"), ("research_question", "research question")):
        if not isinstance(task.get(field), str) or not task[field].strip():
            fail(f"Analysis task has no user-reviewed {label}")
    metadata = variable_metadata(task)
    selected_columns = [task.get("outcome"), *task.get("predictors", []), *task.get("controls", [])]
    missing_labels = [
        str(column)
        for column in selected_columns
        if not isinstance(metadata.get(str(column), {}).get("display_name"), str)
        or not metadata[str(column)]["display_name"].strip()
    ]
    if missing_labels:
        fail("Human-readable variable names are missing for: " + ", ".join(missing_labels))
    if prep.get("data_preparation_executed") is not True:
        fail("Preparation log does not confirm data preparation")
    missingness_screen = prep.get("missingness_bias_screen")
    conclusion_contract = prep.get("missingness_conclusion_contract")
    post_missingness = prep.get("post_preparation_missingness")
    if not isinstance(missingness_screen, dict):
        fail("Preparation log is missing the pre-treatment missingness screen")
    if missingness_screen.get("diagnostic_scope") != "descriptive-sample-composition":
        fail("Preparation log has an invalid missingness-screen scope")
    interpretation = missingness_screen.get("interpretation")
    if not isinstance(interpretation, dict) or interpretation.get("can_prove_no_selection_bias") is not False:
        fail("Missingness screen must explicitly prohibit a no-selection-bias claim")
    if not isinstance(conclusion_contract, dict) or not isinstance(post_missingness, dict):
        fail("Preparation log is missing the missingness conclusion or post-preparation contract")
    if (
        missingness_screen.get("status") == "review-required"
        and conclusion_contract.get("scope") != "analyzed-sample-only"
        and conclusion_contract.get("model_estimate_sensitivity_completed") is not True
    ):
        fail("Missing selected-field data require analyzed-sample-only conclusions or completed model sensitivity evidence")
    if task.get("status") != "approved" or not isinstance(task.get("approval"), dict):
        fail("Analysis task is not backed by a questionnaire approval")
    if (
        spec.get("status") != "approved"
        or spec.get("user_confirmed") is not True
        or not isinstance(spec.get("approval"), dict)
    ):
        fail("Model specification is not explicitly approved")
    if summary.get("status") != "completed" or summary.get("modeling_executed") is not True:
        fail("Model summary does not confirm completed modeling")
    if summary.get("missingness_bias_screen") != missingness_screen:
        fail("Model summary missingness evidence differs from the preparation log")
    if summary.get("missingness_conclusion_contract") != conclusion_contract:
        fail("Model summary missingness conclusion scope differs from the preparation log")
    influence_summary = summary.get("influence_diagnostics")
    if not isinstance(influence_summary, dict) or influence_summary.get("status") not in {
        "available", "not-evaluated"
    }:
        fail("Model summary is missing an explicit influence-assessment status")
    if influence_summary.get("status") == "available":
        if int(influence_summary.get("candidate_count", -1)) != len(influence_rows):
            fail("Influence diagnostic CSV candidate count differs from the model summary")
        if influence_summary.get("automatic_deletion_performed") is not False:
            fail("Influence candidates must not be automatically deleted")
    elif influence_rows:
        fail("Influence rows cannot be reported when assessment status is not-evaluated")
    approved_iia = spec.get("iia_check")
    if not isinstance(approved_iia, dict):
        fail("Approved specification is missing the IIA check")
    if summary.get("iia_check") != iia:
        fail("Model summary and IIA artifact differ")
    if str(spec.get("model_type")) == "multinomial-logistic":
        if iia.get("model_fitting_allowed") is not True:
            fail("Final report cannot use a multinomial model that failed its IIA gate")
        for key in (
            "status", "decision", "reference_category", "outcome_categories",
            "predictor_terms", "evaluated_deletions", "total_deletions",
        ):
            if approved_iia.get(key) != iia.get(key):
                fail("Runtime IIA evidence differs from the approved model")
    elif iia.get("status") != "not-applicable":
        fail("Non-multinomial report has an invalid IIA record")
    approved_proportional_odds = spec.get("proportional_odds_check")
    if not isinstance(approved_proportional_odds, dict):
        fail("Approved specification is missing the proportional-odds check")
    if summary.get("proportional_odds_check") != proportional_odds:
        fail("Model summary and proportional-odds artifact differ")
    if str(spec.get("model_type")) == "ordinal-logistic":
        if proportional_odds.get("model_fitting_allowed") is not True:
            fail("Final report cannot use an ordinal model that failed its proportional-odds gate")
        for key in ("status", "decision", "ordered_categories", "predictor_terms"):
            if approved_proportional_odds.get(key) != proportional_odds.get(key):
                fail("Runtime proportional-odds evidence differs from the approved model")
    elif proportional_odds.get("status") != "not-applicable":
        fail("Non-ordinal report has an invalid proportional-odds record")
    approved_count_dispersion = spec.get("count_dispersion_check")
    if not isinstance(approved_count_dispersion, dict):
        fail("Approved specification is missing the count-dispersion check")
    if summary.get("count_dispersion_check") != count_dispersion:
        fail("Model summary and count-dispersion artifact differ")
    if str(spec.get("model_type")) == "poisson":
        if count_dispersion.get("model_fitting_allowed") is not True:
            fail("Final report cannot use a Poisson model that failed its overdispersion gate")
        for key in ("status", "decision", "test"):
            if approved_count_dispersion.get(key) != count_dispersion.get(key):
                fail("Runtime count-dispersion evidence differs from the approved model")
    elif count_dispersion.get("status") != "not-applicable":
        fail("Non-Poisson report has an invalid count-dispersion record")
    approved_negative_binomial_need = spec.get("negative_binomial_need_check")
    if not isinstance(approved_negative_binomial_need, dict):
        fail("Approved specification is missing the negative-binomial need check")
    if summary.get("negative_binomial_need_check") != negative_binomial_need:
        fail("Model summary and negative-binomial need artifact differ")
    if str(spec.get("model_type")) == "negative-binomial":
        if negative_binomial_need.get("model_fitting_allowed") is not True:
            fail("Final report cannot use a negative-binomial model without supported extra dispersion")
        for key in ("status", "decision", "test"):
            if approved_negative_binomial_need.get(key) != negative_binomial_need.get(key):
                fail("Runtime negative-binomial need evidence differs from the approved model")
    elif negative_binomial_need.get("status") != "not-applicable":
        fail("Non-negative-binomial report has an invalid need-check record")
    approved_zero_inflation = spec.get("zero_inflation_check")
    if not isinstance(approved_zero_inflation, dict):
        fail("Approved specification is missing the excess-zero check")
    if summary.get("zero_inflation_check") != zero_inflation:
        fail("Model summary and excess-zero artifact differ")
    if str(spec.get("model_type")) in {"poisson", "negative-binomial"}:
        if zero_inflation.get("model_fitting_allowed") is not True:
            fail("Final report cannot use an ordinary count model that failed its excess-zero gate")
        for key in ("status", "decision", "test", "model_type"):
            if approved_zero_inflation.get(key) != zero_inflation.get(key):
                fail("Runtime excess-zero evidence differs from the approved model")
    elif zero_inflation.get("status") != "not-applicable":
        fail("Non-count report has an invalid excess-zero record")
    if summary.get("predictive_validation") != predictive_validation:
        fail("Model summary and predictive-validation artifact differ")
    if str(spec.get("goal")) == "prediction":
        if predictive_validation.get("status") != "completed" or predictive_validation.get(
            "model_performance_claim_allowed"
        ) is not True:
            fail("Prediction reports require completed out-of-fold validation")
    elif predictive_validation.get("status") != "not-applicable":
        fail("Non-prediction report has an invalid predictive-validation record")
    if run_log.get("status") != "completed":
        fail("Analysis run log is not completed")
    if spec.get("model_type") != summary.get("model_type"):
        fail("Model type differs between specification and summary")
    categorical = set(map(str, spec.get("categorical_columns", [])))
    reference_map = spec.get("categorical_reference_categories")
    if not isinstance(reference_map, dict) or set(map(str, reference_map)) != categorical:
        fail("Approved specification does not identify every categorical predictor reference")
    if summary.get("categorical_reference_categories") != reference_map:
        fail("Model summary categorical references differ from the approved specification")
    model_type = str(spec.get("model_type"))
    category_support = spec.get("category_support_screen")
    if not isinstance(category_support, dict):
        fail("Approved specification is missing the category-support screen")
    if summary.get("category_support_screen") != category_support:
        fail("Model summary category-support evidence differs from the approved specification")
    if category_support.get("model_fitting_allowed") is not True:
        fail("Final report cannot use a model with unresolved category-support risk")
    expected_support_rows = flatten_category_support_rows(category_support)
    support_count_columns = [
        f"outcome_count:{category}"
        for category in category_support.get("outcome_categories", [])
    ]
    def support_key(row: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(row.get("factor")),
            str(row.get("level")),
            str(row.get("total")),
            *(str(row.get(column)) for column in support_count_columns),
            str(row.get("risk_codes") or ""),
            str(row.get("blocking")).lower(),
        )
    expected_support_keys = {support_key(row) for row in expected_support_rows}
    file_support_keys = {support_key(row) for row in category_support_rows}
    if expected_support_keys != file_support_keys:
        fail("Category-support CSV differs from the approved model specification")
    if model_type == "logistic":
        if spec.get("positive_class") is None or spec.get("outcome_reference_class") is None:
            fail("Approved logistic specification is missing target/reference outcome classes")
        if (
            summary.get("positive_class") != spec.get("positive_class")
            or summary.get("outcome_reference_class")
            != spec.get("outcome_reference_class")
        ):
            fail("Model summary logistic classes differ from the approved specification")
    if model_type in SUPPORTED_MODEL_TYPE_SET:
        continuous_columns = set(map(str, [*spec.get("predictors", []), *spec.get("controls", [])])) - categorical
        forms = spec.get("continuous_functional_forms")
        if not isinstance(forms, dict) or set(map(str, forms)) != continuous_columns:
            fail(f"Approved {model_type.upper()} specification is missing continuous functional forms")
        if summary.get("continuous_functional_forms") != forms:
            fail("Model summary continuous functional forms differ from the approved specification")
        summary_shape_tests = summary.get("continuous_shape_tests")
        if not isinstance(summary_shape_tests, list):
            fail("Model summary is missing continuous shape tests")
        if {str(item.get("variable")) for item in summary_shape_tests if isinstance(item, dict)} != {
            str(item.get("variable")) for item in shape_tests
        } or {str(item.get("variable")) for item in shape_tests} != continuous_columns:
            fail("Continuous shape-test file differs from the model summary/specification")
    if model_type == "multinomial-logistic" and (
        spec.get("reference_class") is None
        or summary.get("reference_class") != spec.get("reference_class")
    ):
        fail("Multinomial result reference differs from the approved specification")
    multiplicity = summary.get("multiplicity")
    if not isinstance(multiplicity, dict):
        fail("Model summary is missing the multiplicity-control record")
    if multiplicity.get("coefficient_method") != "Benjamini-Hochberg FDR":
        fail("Model summary does not use the approved coefficient multiplicity method")
    collinearity = summary.get("collinearity")
    if not isinstance(collinearity, dict) or collinearity.get("status") not in {
        "clear",
        "review",
        "severe",
    }:
        fail("Model summary is missing a valid collinearity decision record")
    summary_factor_tests = multiplicity.get("categorical_omnibus_tests", [])
    if not isinstance(summary_factor_tests, list):
        fail("Model summary categorical omnibus record is invalid")
    summary_factors = {
        str(item.get("factor"))
        for item in summary_factor_tests
        if isinstance(item, dict)
    }
    file_factors = {str(item.get("factor")) for item in factor_tests}
    if summary_factors != file_factors or summary_factors != categorical:
        fail("Categorical omnibus test file differs from the model summary/specification")
    for row in results:
        if row.get("term") == "const" or row.get("term_type") not in {
            None,
            "",
            "coefficient",
        }:
            continue
        if to_float(row.get("p_value_adjusted_bh")) is None:
            fail(f"Adjusted p-value is missing for model term {row.get('term')!r}")
        if to_bool(row.get("multiplicity_supported")) is None:
            fail(f"Multiplicity decision is missing for model term {row.get('term')!r}")
        restricted = to_bool(row.get("collinearity_restricted"))
        supported = to_bool(row.get("interpretation_supported"))
        if restricted is None or supported is None:
            fail(f"Collinearity interpretation decision is missing for {row.get('term')!r}")
        if supported != (multiplicity_supported(row) and not restricted):
            shape_restricted = to_bool(row.get("shape_basis_restricted")) is True
            if supported != (multiplicity_supported(row) and not restricted and not shape_restricted):
                fail(f"Interpretation decision is inconsistent for {row.get('term')!r}")
        column, level = split_result_term(str(row.get("term") or ""), spec)
        if level is not None and column in categorical:
            if (
                to_float(row.get("factor_omnibus_p_value")) is None
                or to_float(row.get("factor_omnibus_p_value_adjusted_bh")) is None
            ):
                fail(f"Categorical omnibus result is missing for {column!r}")
    if not results:
        fail("Model results table is empty")
    expected_hashes = {
        "data_sha256": sha256_file(cleaned_data),
        "preparation_log_sha256": sha256_file(prep_path),
        "model_specification_sha256": sha256_file(spec_path),
    }
    for key, actual in expected_hashes.items():
        if run_log.get(key) != actual:
            fail(f"Analysis run log hash mismatch for {key}")


def profile_summary(profile: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = profile.get("profiles", [])
    first = profiles[0] if profiles and isinstance(profiles[0], dict) else {}
    columns = first.get("columns", []) if isinstance(first.get("columns", []), list) else []
    return first, [item for item in columns if isinstance(item, dict)]


def variable_metadata(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = task.get("variable_metadata", {})
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
    if isinstance(raw, list):
        return {
            str(item.get("column")): item
            for item in raw
            if isinstance(item, dict) and item.get("column")
        }
    return {}


def display_name(task: dict[str, Any], column: Any) -> str:
    name = str(column or "")
    return str(variable_metadata(task).get(name, {}).get("display_name") or name)


def display_unit(task: dict[str, Any], column: Any) -> str:
    name = str(column or "")
    return str(variable_metadata(task).get(name, {}).get("unit") or "")


def dataset_intro(profile: dict[str, Any], task: dict[str, Any]) -> str:
    supplied = task.get("dataset_summary")
    if isinstance(supplied, str) and supplied.strip():
        return supplied.strip()
    sheet, columns = profile_summary(profile)
    rows = sheet.get("row_count", "若干")
    unit = task.get("unit_of_analysis") or "一条观测记录"
    return f"这是一份包含 {rows} 条记录、{len(columns)} 个字段的数据集，每条记录代表{unit}。"


def research_question(task: dict[str, Any]) -> str:
    supplied = task.get("research_question")
    if isinstance(supplied, str) and supplied.strip():
        return supplied.strip()
    outcome = display_name(task, task.get("outcome", "研究结果"))
    factors = [display_name(task, value) for value in task.get("predictors", [])]
    factor_text = "、".join(factors) or "所关注的因素"
    return f"{factor_text}与{outcome}之间有什么关系？"


def split_result_term(
    term: str, spec: dict[str, Any]
) -> tuple[str, str | None]:
    columns = [*spec.get("predictors", []), *spec.get("controls", [])]
    for column in sorted(map(str, columns), key=len, reverse=True):
        if term == column:
            return column, None
        prefix = f"{column}_"
        if term.startswith(prefix):
            return column, term[len(prefix):]
    return term, None


def meaningful_increment(task: dict[str, Any], column: str) -> tuple[float, str]:
    metadata = variable_metadata(task).get(column, {})
    increment = to_float(metadata.get("interpretation_increment")) or 1.0
    label = metadata.get("interpretation_increment_label")
    if isinstance(label, str) and label.strip():
        return increment, label.strip()
    unit = str(metadata.get("unit") or "个单位")
    return increment, f"{fmt_number(increment)}{unit}"


def effect_explanation(
    row: dict[str, str], spec: dict[str, Any], task: dict[str, Any]
) -> tuple[str, str, str]:
    term = str(row.get("term") or "")
    column, level = split_result_term(term, spec)
    metadata = variable_metadata(task).get(column, {})
    factor = display_name(task, column)
    outcome = display_name(task, spec.get("outcome"))
    outcome_unit = display_unit(task, spec.get("outcome"))
    estimate = to_float(row.get("estimate"))
    low = to_float(row.get("ci_low"))
    high = to_float(row.get("ci_high"))
    if estimate is None or low is None or high is None:
        return factor, "当前结果缺少完整估计值或置信区间，不能进行可靠解释。", "信息不完整"

    clear_direction = low > 0 or high < 0
    status = "估计方向较明确" if clear_direction else "方向仍有不确定性"
    if clear_direction and not multiplicity_supported(row):
        status = "多重比较校正后证据不足"
    elif clear_direction and not interpretation_supported(row):
        status = "受严重共线性限制，不作独立作用解释"
    model_type = str(spec.get("model_type"))
    if level is not None:
        increment = 1.0
        reference_map = spec.get("categorical_reference_categories", {})
        comparison = str(
            reference_map.get(column)
            if isinstance(reference_map, dict) and reference_map.get(column) is not None
            else metadata.get("reference_category") or "参照组"
        )
        change_label = f"{factor}为“{level}”而不是“{comparison}”"
        title = f"{factor}：{level} 与 {comparison}"
    else:
        increment, increment_label = meaningful_increment(task, column)
        change_label = f"{factor}每增加{increment_label}"
        title = factor

    scaled_estimate = estimate * increment
    scaled_low = min(low * increment, high * increment)
    scaled_high = max(low * increment, high * increment)
    if model_type == "ols":
        direction = "高" if scaled_estimate > 0 else ("低" if scaled_estimate < 0 else "接近")
        unit = outcome_unit
        if clear_direction:
            text = (
                f"控制其他纳入因素后，{change_label}对应的{outcome}平均{direction}约"
                f"{fmt_number(abs(scaled_estimate), 2)}{unit}，区间为"
                f"{fmt_number(scaled_low, 2)}～{fmt_number(scaled_high, 2)}{unit}。"
            )
        else:
            text = (
                f"点估计为 {fmt_number(scaled_estimate, 2)}{unit}，但区间"
                f"{fmt_number(scaled_low, 2)}～{fmt_number(scaled_high, 2)}{unit}包含0；"
                "现有数据不足以判断稳定的升高或降低方向。"
            )
        return title, text, status

    ratio = math.exp(scaled_estimate)
    ratio_low = math.exp(scaled_low)
    ratio_high = math.exp(scaled_high)
    if not clear_direction:
        uncertain_title = title
        if model_type == "multinomial-logistic":
            category = str(row.get("outcome_category") or "当前类别")
            reference = str(
                row.get("reference_class")
                or spec.get("reference_class")
                or "参照类别"
            )
            uncertain_title = f"{title}｜{category} 对 {reference}"
        text = (
            f"点估计为 {fmt_number(ratio)} 倍，区间 "
            f"{fmt_number(ratio_low)}～{fmt_number(ratio_high)} 倍包含1；"
            "现有数据不足以判断稳定的升高或降低方向。"
        )
        return uncertain_title, text, status
    if model_type == "logistic":
        direction = "更高" if ratio > 1 else "更低"
        text = (
            f"在其他已纳入因素相同的情况下，{change_label}时，{outcome}发生的优势约为原来的 "
            f"{fmt_number(ratio)} 倍（区间 {fmt_number(ratio_low)}～{fmt_number(ratio_high)} 倍），"
            f"即相对{direction}；这不是概率直接增加或减少的百分比。"
        )
        return title, text, status

    if model_type == "multinomial-logistic":
        category = str(row.get("outcome_category") or "当前类别")
        reference = str(row.get("reference_class") or spec.get("reference_class") or "参照类别")
        direction = "更高" if ratio > 1 else "更低"
        text = (
            f"在其他已纳入因素相同的情况下，{change_label}时，结果属于“{category}”而不是"
            f"“{reference}”的相对风险约为原来的 {fmt_number(ratio)} 倍"
            f"（区间 {fmt_number(ratio_low)}～{fmt_number(ratio_high)} 倍），即相对{direction}。"
        )
        return f"{title}｜{category} 对 {reference}", text, status

    if model_type == "ordinal-logistic":
        direction = "更高" if ratio > 1 else "更低"
        text = (
            f"在其他已纳入因素相同的情况下，{change_label}时，{outcome}处于更高等级的累计优势"
            f"约为原来的 {fmt_number(ratio)} 倍"
            f"（区间 {fmt_number(ratio_low)}～{fmt_number(ratio_high)} 倍），即相对{direction}。"
        )
        return title, text, status

    direction = "更高" if ratio > 1 else "更低"
    text = (
        f"在其他已纳入因素相同的情况下，{change_label}时，{outcome}的预期计数约为原来的 "
        f"{fmt_number(ratio)} 倍（区间 {fmt_number(ratio_low)}～{fmt_number(ratio_high)} 倍），即相对{direction}。"
    )
    return title, text, status


def result_for_primary_predictor(
    results: list[dict[str, str]], predictor: str
) -> dict[str, str] | None:
    for row in results:
        term = row.get("term", "")
        if row.get("term_type") in {None, "", "coefficient"} and (
            term == predictor or term.startswith(f"{predictor}_")
        ):
            return row
    for row in results:
        if row.get("term") != "const" and row.get("term_type") in {None, "", "coefficient"}:
            return row
    return None


def build_core_conclusion(
    spec: dict[str, Any], results: list[dict[str, str]], task: dict[str, Any], prep: dict[str, Any] | None = None
) -> str:
    outcome = display_name(task, spec.get("outcome", "所关注的结果"))
    predictors = [str(value) for value in spec.get("predictors", [])]
    if not predictors:
        return "分析已经完成，但没有可解释的主要因素结果，应返回模型规格阶段检查变量设置。"
    categorical = set(map(str, spec.get("categorical_columns", [])))
    alpha = 1.0 - float(spec.get("confidence_level", 0.95))
    positive: list[str] = []
    negative: list[str] = []
    categorical_supported: list[str] = []
    nonlinear_supported: list[str] = []
    uncertain: list[str] = []
    for predictor in predictors:
        matching = [
            row
            for row in results
            if row.get("term_type") in {None, "", "coefficient"}
            and (
                row.get("term") == predictor
                or str(row.get("term") or "").startswith(f"{predictor}_")
            )
        ]
        label = display_name(task, predictor)
        if not matching:
            uncertain.append(label)
            continue
        if predictor in categorical:
            omnibus_adjusted = to_float(
                matching[0].get("factor_omnibus_p_value_adjusted_bh")
            )
            collinearity_restricted = any(
                to_bool(row.get("collinearity_restricted")) is True
                for row in matching
            )
            if (
                omnibus_adjusted is not None
                and omnibus_adjusted < alpha
                and not collinearity_restricted
            ):
                categorical_supported.append(label)
            else:
                uncertain.append(label)
            continue
        form_spec = spec.get("continuous_functional_forms", {}).get(predictor, {})
        if isinstance(form_spec, dict) and form_spec.get("form") in {
            "quadratic", "restricted-cubic-spline"
        }:
            overall_adjusted = to_float(
                matching[0].get("continuous_overall_p_value_adjusted_bh")
            )
            if overall_adjusted is not None and overall_adjusted < alpha:
                nonlinear_supported.append(label)
            else:
                uncertain.append(label)
            continue
        row = matching[0]
        estimate = to_float(row.get("estimate"))
        low = to_float(row.get("ci_low"))
        high = to_float(row.get("ci_high"))
        if (
            estimate is None
            or low is None
            or high is None
            or (low <= 0 <= high)
            or not interpretation_supported(row)
        ):
            uncertain.append(label)
        elif estimate > 0:
            positive.append(label)
        else:
            negative.append(label)
    clauses: list[str] = []
    if positive:
        clauses.append(f"{'、'.join(positive)}与{outcome}呈正向条件关联")
    if negative:
        clauses.append(f"{'、'.join(negative)}与{outcome}呈负向条件关联")
    if categorical_supported:
        clauses.append(
            f"{'、'.join(categorical_supported)}的整体检验支持其与{outcome}存在条件关联，"
            "具体类别方向需分别查看"
        )
    if nonlinear_supported:
        clauses.append(
            f"{'、'.join(nonlinear_supported)}的整体检验支持其与{outcome}存在条件关联，"
            "关系方向可能随取值改变，应以调整后预测曲线解释"
        )
    if uncertain:
        clauses.append(f"{'、'.join(uncertain)}未通过完整证据门槛")
    scope = ((prep or {}).get("missingness_conclusion_contract") or {}).get("scope")
    opening = "在实际进入模型的分析样本中，" if scope == "analyzed-sample-only" else "在本样本中，"
    return opening + "在同时考虑其他已纳入因素后，" + "；".join(clauses) + "；这些结果描述统计关联，不能直接解释为因果关系。"


def build_plain_findings_html(
    results: list[dict[str, str]], spec: dict[str, Any], task: dict[str, Any]
) -> str:
    nonlinear_variables = [
        str(column)
        for column, item in spec.get("continuous_functional_forms", {}).items()
        if isinstance(item, dict)
        and item.get("form") in {"quadratic", "restricted-cubic-spline"}
    ] if isinstance(spec.get("continuous_functional_forms"), dict) else []
    nonlinear_note = (
        '<div class="callout"><strong>非线性连续变量：</strong>'
        + esc("、".join(display_name(task, column) for column in nonlinear_variables))
        + "采用整体曲线解释；单个基函数系数不作为现实中的独立效应。</div>"
        if nonlinear_variables
        else ""
    )
    coefficient_rows = [
        row
        for row in results
        if row.get("term") != "const"
        and row.get("term_type") in {None, "", "coefficient"}
        and to_bool(row.get("nonlinear_basis_term")) is not True
        and row.get("continuous_functional_form") not in {
            "quadratic", "restricted-cubic-spline"
        }
    ]
    if not coefficient_rows:
        return nonlinear_note + '<p class="lead">其余结果没有可逐项解释的非截距模型项。</p>'

    def row_label(row: dict[str, str]) -> str:
        title, _explanation, _status = effect_explanation(row, spec, task)
        return title

    def short_group(labels: list[str]) -> str:
        shown = labels[:MAIN_SUMMARY_LIMIT]
        suffix = f"等 {len(labels)} 项" if len(labels) > MAIN_SUMMARY_LIMIT else ""
        return "、".join(shown) + suffix

    positive: list[str] = []
    negative: list[str] = []
    uncertain: list[str] = []
    for row in coefficient_rows:
        estimate = to_float(row.get("estimate"))
        low = to_float(row.get("ci_low"))
        high = to_float(row.get("ci_high"))
        label = row_label(row)
        if (
            estimate is None
            or low is None
            or high is None
            or low <= 0 <= high
            or not interpretation_supported(row)
        ):
            uncertain.append(label)
        elif estimate > 0:
            positive.append(label)
        else:
            negative.append(label)

    summary_items: list[str] = []
    if positive:
        summary_items.append(f"方向较明确的正向关联主要见于{short_group(positive)}")
    if negative:
        summary_items.append(f"方向较明确的负向关联主要见于{short_group(negative)}")
    if uncertain:
        summary_items.append(
            f"另有 {len(uncertain)} 项未同时通过区间、多重比较与共线性解释门槛，暂不判断稳定方向"
        )

    def finding_sort_key(row: dict[str, str]) -> tuple[bool, bool, float]:
        adjusted_p = to_float(row.get("p_value_adjusted_bh"))
        return (
            not interpretation_supported(row),
            adjusted_p is None,
            adjusted_p if adjusted_p is not None else math.inf,
        )

    ordered_rows = sorted(
        coefficient_rows,
        key=finding_sort_key,
    )
    key_rows: list[str] = []
    ratio_model = str(spec.get("model_type")) != "ols"
    for row in ordered_rows[: MAIN_SUMMARY_LIMIT * 2]:
        title, _explanation, status = effect_explanation(row, spec, task)
        low = to_float(row.get("ci_low"))
        high = to_float(row.get("ci_high"))
        if ratio_model:
            estimate = to_float(row.get("exp_estimate"))
            if estimate is None:
                raw_estimate = to_float(row.get("estimate"))
                estimate = math.exp(raw_estimate) if raw_estimate is not None else None
            interval = (
                f"{fmt_number(math.exp(low))}～{fmt_number(math.exp(high))}"
                if low is not None and high is not None
                else "—"
            )
        else:
            estimate = to_float(row.get("estimate"))
            interval = (
                f"{fmt_number(low)}～{fmt_number(high)}"
                if low is not None and high is not None
                else "—"
            )
        key_rows.append(
            "<tr>"
            f"<td>{esc(title)}</td>"
            f'<td class="num">{fmt_number(estimate)}</td>'
            f'<td class="num">{esc(interval)}</td>'
            f'<td class="num">{esc(fmt_p(row.get("p_value_adjusted_bh")))}</td>'
            f"<td>{esc(status)}</td>"
            "</tr>"
        )
    effect_label = "比值效应" if ratio_model else "系数"
    return nonlinear_note + (
        '<div class="finding-summary">'
        f'<p>{esc("；".join(summary_items))}。</p>'
        "</div>"
        '<div class="table-wrap finding-table"><table><thead><tr>'
        f'<th>关键项目</th><th class="num">{effect_label}</th>'
        '<th class="num">置信区间</th><th class="num">BH校正p值</th><th>判断</th></tr></thead>'
        f'<tbody>{"".join(key_rows)}</tbody></table></div>'
        f'<p class="detail-reference">正文仅列出最多 {MAIN_SUMMARY_LIMIT * 2} 个关键模型项；'
        "完整参数结果以紧凑表格列于技术与诊断明细。</p>"
    )


def build_dataset_html(
    profile: dict[str, Any], task: dict[str, Any], prep: dict[str, Any], spec: dict[str, Any], summary: dict[str, Any]
) -> str:
    sheet, _columns = profile_summary(profile)
    source_file = profile.get("source_file", prep.get("source_file", "—"))
    original_rows = sheet.get("row_count", prep.get("input_rows", "—"))
    original_columns = sheet.get("column_count", prep.get("input_columns", "—"))
    processed_rows = prep.get("output_rows", spec.get("complete_case_rows", "—"))
    rows_used = summary.get("rows_used", "—")
    metrics = "".join(
        [
            f'<div class="metric"><span class="metric-label">原始记录数</span><span class="metric-value">{esc(original_rows)}</span></div>',
            f'<div class="metric"><span class="metric-label">原始变量数</span><span class="metric-value">{esc(original_columns)}</span></div>',
            f'<div class="metric"><span class="metric-label">处理后记录数</span><span class="metric-value">{esc(processed_rows)}</span></div>',
            f'<div class="metric"><span class="metric-label">模型使用记录数</span><span class="metric-value">{esc(rows_used)}</span></div>',
        ]
    )
    return (
        f'<p class="lead">{esc(dataset_intro(profile, task))}</p>'
        '<div class="question-box">'
        f'<p class="question-text">{esc(research_question(task))}</p></div>'
        f'<p>分析数据文件为 <code>{esc(Path(str(source_file)).name)}</code>；'
        f'每条记录代表“{esc(spec.get("unit_of_analysis") or task.get("unit_of_analysis") or "一个观测对象")}”。</p>'
        f'<div class="metric-grid">{metrics}</div>'
    )


def build_variables_html(task: dict[str, Any]) -> str:
    metadata = variable_metadata(task)
    roles: list[tuple[str, str]] = [(str(task.get("outcome")), "本次关注的结果")]
    roles.extend((str(value), "重点关注的因素") for value in task.get("predictors", []))
    roles.extend((str(value), "同时考虑的其他因素") for value in task.get("controls", []))
    variable_types = {
        str(item.get("column")): str(item.get("inferred_type") or "—")
        for item in task.get("variables", [])
        if isinstance(item, dict) and item.get("column")
    }
    rows: list[str] = []
    for column, purpose in roles:
        meta = metadata.get(column, {})
        unit = str(meta.get("unit") or "—")
        description = str(meta.get("description") or "—")
        rows.append(
            "<tr>"
            f'<td>{esc(display_name(task, column))}</td>'
            f'<td class="term">{esc(column)}</td>'
            f'<td>{esc(purpose)}</td>'
            f'<td>{esc(unit)}</td>'
            f'<td>{esc(variable_types.get(column, "—"))}</td>'
            f'<td>{esc(description)}</td>'
            "</tr>"
        )
    return (
        '<p class="lead">下表列出本次分析使用的字段、中文名称、单位和用途。原始字段名仅用于核查数据与代码。</p>'
        '<div class="table-wrap"><table><thead><tr><th>中文名称</th><th>原始字段</th><th>在本次分析中的用途</th><th>单位</th><th>数据类型</th><th>说明</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def sensitive_field_review(
    profile: dict[str, Any], task: dict[str, Any]
) -> tuple[list[str], list[str]]:
    sheet, columns = profile_summary(profile)
    candidates: dict[str, str] = {
        str(column.get("column")): str(column.get("column"))
        for column in columns
        if column.get("sensitive_name_candidate")
    }
    review = sheet.get("sensitive_review", {})
    unresolved = (
        {
            str(column)
            for column in review.get("opaque_columns_requiring_semantic_review", [])
            if isinstance(column, str)
        }
        if isinstance(review, dict)
        else set()
    )
    metadata = task.get("variable_metadata", {})
    if isinstance(metadata, dict):
        for raw_column, value in metadata.items():
            if not isinstance(value, dict) or value.get("user_confirmed") is not True:
                continue
            raw_name = str(raw_column)
            unresolved.discard(raw_name)
            semantic_text = " ".join(
                str(value.get(key, "")) for key in ("display_name", "meaning")
            )
            if SENSITIVE_SEMANTIC_RE.search(semantic_text):
                display = str(value.get("display_name") or raw_name)
                candidates[raw_name] = (
                    f"{display}（{raw_name}）" if display != raw_name else raw_name
                )
    return list(candidates.values()), sorted(unresolved)


def missingness_outcome_text(summary: dict[str, Any]) -> str:
    if summary.get("kind") == "categorical":
        parts = []
        for row in summary.get("distribution", []):
            if not isinstance(row, dict):
                continue
            proportion = row.get("proportion")
            rate = f"{float(proportion):.1%}" if isinstance(proportion, (int, float)) else "—"
            parts.append(f"{row.get('value')}：{row.get('count', 0)}（{rate}）")
        return "；".join(parts) or "无可用结果记录"
    mean = summary.get("mean")
    median = summary.get("median")
    return (
        f"n={summary.get('count', 0)}，均值={float(mean):.3g}，中位数={float(median):.3g}"
        if isinstance(mean, (int, float)) and isinstance(median, (int, float))
        else "无可用结果记录"
    )


def build_quality_html(
    profile: dict[str, Any], prep: dict[str, Any], task: dict[str, Any]
) -> str:
    sheet, columns = profile_summary(profile)
    missing_columns = [column for column in columns if (column.get("missing_count") or 0) > 0]
    missing_columns.sort(
        key=lambda column: float(column.get("missing_rate") or 0), reverse=True
    )
    missing_cells = sum(int(column.get("missing_count") or 0) for column in columns)
    screen = prep.get("missingness_bias_screen", {})
    post_missingness = prep.get("post_preparation_missingness", {})
    conclusion_contract = prep.get("missingness_conclusion_contract", {})
    sensitive, unresolved_sensitive = sensitive_field_review(profile, task)
    actions = prep.get("actions", [])
    action_rows: list[str] = []
    normalized_actions = actions if isinstance(actions, list) else []
    for action in normalized_actions[:MAIN_DETAIL_LIMIT]:
        if not isinstance(action, dict):
            continue
        details = action.get("details", {}) if isinstance(action.get("details"), dict) else {}
        action_rows.append(
            "<tr>"
            f'<td class="term">{esc(action.get("decision_id"))}</td>'
            f'<td>{esc(CHOICE_LABELS.get(str(action.get("selected_option")), action.get("selected_option")))}</td>'
            f'<td>{esc(details.get("column", "—"))}</td>'
            f'<td class="num">{esc(action.get("affected_row_count", 0))}</td>'
            f'<td class="num">{esc(action.get("rows_after", "—"))}</td>'
            "</tr>"
        )
    if len(missing_columns) > MAIN_DETAIL_LIMIT:
        top_missing = "、".join(
            f'{column.get("column")}（{float(column.get("missing_rate") or 0):.1%}）'
            for column in missing_columns[:MAIN_SUMMARY_LIMIT]
        )
        missing_text = (
            f"共 {len(missing_columns)} 个字段存在缺失；缺失率较高的字段包括 {top_missing}。"
            "完整字段清单见末尾“技术与诊断明细”。"
        )
    else:
        missing_text = "、".join(
            f'{column.get("column")}（{float(column.get("missing_rate") or 0):.1%}）'
            for column in missing_columns
        ) or "未检测到缺失字段"
    deferred_items = [
        esc(item.get("reason"))
        for item in prep.get("deferred_actions", [])
        if isinstance(item, dict)
    ]
    action_reference = (
        f'<p class="detail-reference">正文仅展示前 {MAIN_DETAIL_LIMIT} 项处理动作；'
        '完整处理清单见末尾“技术与诊断明细”。</p>'
        if len(normalized_actions) > MAIN_DETAIL_LIMIT
        else ""
    )
    deferred_reference = (
        f'<p class="detail-reference">正文仅展示前 {MAIN_DETAIL_LIMIT} 项延后事项；'
        '完整清单见末尾“技术与诊断明细”。</p>'
        if len(deferred_items) > MAIN_DETAIL_LIMIT
        else ""
    )
    sensitive_callout = ""
    if sensitive:
        if len(sensitive) > MAIN_DETAIL_LIMIT:
            sensitive_text = (
                f"共 {len(sensitive)} 个潜在敏感字段，包括"
                f'{esc("、".join(map(str, sensitive[:MAIN_SUMMARY_LIMIT])))}等。'
                "完整字段清单见末尾“技术与诊断明细”。"
            )
        else:
            sensitive_text = f'数据体检将 {esc("、".join(map(str, sensitive)))} 标记为潜在敏感字段。'
        sensitive_callout = (
            '<div class="callout"><strong>共享提醒：</strong>'
            f'{sensitive_text}'
            '处理后数据文件可能仍包含这些字段，分享报告目录前应再次审查。</div>'
        )
    if unresolved_sensitive:
        unresolved_text = "、".join(unresolved_sensitive[:MAIN_SUMMARY_LIMIT])
        sensitive_callout += (
            '<div class="callout"><strong>仍需核对：</strong>'
            f'另有 {len(unresolved_sensitive)} 个字段使用无含义列名（如 {esc(unresolved_text)}），'
            '不能据此断言它们不敏感；需要结合用户确认的字段含义完成审查。</div>'
        )
    sensitive_metric = (
        f"待核对（{len(unresolved_sensitive)}）"
        if unresolved_sensitive
        else str(len(sensitive))
    )
    missingness_callout = (
        '<div class="callout"><strong>缺失值对样本构成的影响：</strong>'
        f"若所选字段统一采用完整案例处理，将保留 {esc(screen.get('complete_case_rows', '—'))} 行，"
        f"排除 {esc(screen.get('complete_case_excluded_rows', '—'))} 行"
        f"（{float(screen.get('complete_case_excluded_rate', 0) or 0):.1%}）。"
        f"保留组结果构成：{esc(missingness_outcome_text(screen.get('complete_case_outcome_retained', {})))}；"
        f"排除组结果构成：{esc(missingness_outcome_text(screen.get('complete_case_outcome_excluded', {})))}。"
        "这是处理前的描述性检查，不能证明缺失是随机的，也没有替代不同缺失处理方法下的模型估计敏感性分析。"
        f"因此本报告把结论范围限定为：{esc(conclusion_contract.get('scope', '未记录'))}。"
        '</div>'
        if screen.get("status") == "review-required"
        else '<div class="callout"><strong>缺失值检查：</strong>所选分析字段未发现原始缺失。</div>'
    )
    return (
        '<div class="metric-grid">'
        f'<div class="metric"><span class="metric-label">重复记录</span><span class="metric-value">{esc(sheet.get("duplicate_row_count", 0))}</span></div>'
        f'<div class="metric"><span class="metric-label">原始缺失单元格</span><span class="metric-value">{missing_cells}</span></div>'
        f'<div class="metric"><span class="metric-label">全空记录</span><span class="metric-value">{esc(sheet.get("fully_empty_row_count", 0))}</span></div>'
        f'<div class="metric"><span class="metric-label">潜在敏感字段</span><span class="metric-value">{esc(sensitive_metric)}</span></div>'
        '</div>'
        f'<p><strong>原始数据缺失情况：</strong>{esc(missing_text)}</p>'
        f'<p><strong>处理后所选字段：</strong>仍有 {esc(post_missingness.get("rows_with_any_selected_missingness", "—"))} 行至少缺失一项；建模程序如再做完整案例排除，须在模型摘要中另行记录。</p>'
        f'{missingness_callout}'
        f'{sensitive_callout}'
        '<h3>已执行的数据处理</h3>'
        '<div class="table-wrap"><table><thead><tr><th>决定</th><th>用户选择</th><th>字段</th><th class="num">影响记录数</th><th class="num">处理后记录数</th></tr></thead>'
        f'<tbody>{"".join(action_rows) or "<tr><td colspan=\"5\">无处理动作记录。</td></tr>"}</tbody></table></div>'
        f'{action_reference}'
        '<h3>延后事项</h3>'
        f'{html_list(deferred_items[:MAIN_DETAIL_LIMIT])}'
        f'{deferred_reference}'
    )


def eda_variable_roles(
    task: dict[str, Any], spec: dict[str, Any], selected: list[str]
) -> dict[str, str]:
    """Return final model roles; early task-card guesses must not override them."""
    roles = {column: "continuous" for column in selected}
    outcome = str(spec.get("outcome"))
    model_type = str(spec.get("model_type"))

    for column in spec.get("categorical_columns", []):
        name = str(column)
        if name in roles and name != outcome:
            roles[name] = "nominal"

    metadata = task.get("variable_metadata", {})
    if isinstance(metadata, dict):
        for name, value in metadata.items():
            if str(name) not in roles or not isinstance(value, dict):
                continue
            if roles[str(name)] != "continuous" and isinstance(value.get("category_order"), list):
                roles[str(name)] = "ordinal"
            elif roles[str(name)] != "continuous" and isinstance(value.get("category_meanings"), dict):
                roles[str(name)] = "nominal"
    if model_type == "logistic":
        roles[outcome] = "binary"
    elif model_type == "multinomial-logistic":
        roles[outcome] = "nominal"
    elif model_type == "ordinal-logistic":
        roles[outcome] = "ordinal"
    return roles


def build_eda_artifacts(
    cleaned_data: Path,
    task: dict[str, Any],
    spec: dict[str, Any],
    output_figures: Path,
    overwrite: bool,
) -> tuple[str, list[str]]:
    matplotlib_cache = Path(tempfile.gettempdir()) / "inno-agent-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        fail(f"EDA generation requires pandas, numpy, and matplotlib: {exc}")

    dataframe = None
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            dataframe = pd.read_csv(cleaned_data, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if dataframe is None:
        fail(f"Cannot decode cleaned data for EDA: {last_error}")

    selected = [
        str(value)
        for value in [spec.get("outcome"), *spec.get("predictors", []), *spec.get("controls", [])]
        if value is not None and str(value) in dataframe.columns
    ]
    selected = list(dict.fromkeys(selected))
    if not selected:
        fail("No selected analysis variables are present in cleaned data for EDA")

    variable_roles = eda_variable_roles(task, spec, selected)
    numeric_series: dict[str, Any] = {}
    categorical_columns: list[str] = []
    summary_rows: list[str] = []

    def category_levels_for(column: str, series: Any) -> list[str]:
        observed = list(dict.fromkeys(series.dropna().astype(str).tolist()))
        approved: list[str] = []
        if column == str(spec.get("outcome")) and isinstance(
            spec.get("outcome_categories"), list
        ):
            approved = [str(value) for value in spec["outcome_categories"]]
        metadata = task.get("variable_metadata", {})
        item = metadata.get(column, {}) if isinstance(metadata, dict) else {}
        if not approved and isinstance(item, dict) and isinstance(
            item.get("category_order"), list
        ):
            approved = [str(value) for value in item["category_order"]]
        if approved:
            observed_set = set(observed)
            return [value for value in approved if value in observed_set]
        counts = series.dropna().astype(str).value_counts()
        return [str(value) for value in counts.index]
    for column in selected:
        original = dataframe[column]
        numeric = pd.to_numeric(original, errors="coerce")
        observed = int(original.notna().sum())
        numeric_observed = int(numeric.notna().sum())
        is_numeric = (
            variable_roles[column] == "continuous"
            and observed > 0
            and numeric_observed / observed >= 0.8
        )
        missing = int(original.isna().sum())
        if is_numeric:
            numeric_series[column] = numeric
            valid = numeric.dropna()
            if len(valid):
                counts = valid.value_counts()
                top_count = int(counts.iloc[0])
                modes = sorted(float(value) for value in counts[counts == top_count].index)
                mode_text = (
                    f"众数 {fmt_number(modes[0], 2)}（{top_count} 条）"
                    if len(modes) == 1
                    else f"众数不唯一（{len(modes)} 个值并列，各 {top_count} 条）"
                )
                summary = (
                    f"中位数 {fmt_number(valid.median(), 2)}；{mode_text}；"
                    f"四分位数 {fmt_number(valid.quantile(0.25), 2)}～{fmt_number(valid.quantile(0.75), 2)}；"
                    f"范围 {fmt_number(valid.min(), 2)}～{fmt_number(valid.max(), 2)}"
                )
            else:
                summary = "无有效数值"
            kind = "连续数值"
        else:
            categorical_columns.append(column)
            counts = original.dropna().astype(str).value_counts()
            levels = category_levels_for(column, original)
            counts = counts.reindex(levels).dropna()
            summary = (
                f"{len(counts)} 个类别；众数为“{counts.index[0]}”（{int(counts.iloc[0])} 条，{int(counts.iloc[0]) / max(observed, 1):.1%}）"
                if len(counts)
                else "无有效类别"
            )
            kind = {
                "binary": "二分类",
                "nominal": "无序多分类",
                "ordinal": "有序多分类",
            }.get(variable_roles[column], "分类")
        summary_rows.append(
            "<tr>"
            f'<td>{esc(display_name(task, column))}</td>'
            f'<td>{kind}</td><td class="num">{observed}</td><td class="num">{missing}</td>'
            f'<td>{esc(summary)}</td></tr>'
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9aabb2",
            "axes.labelcolor": "#344f5b",
            "xtick.color": "#53676f",
            "ytick.color": "#53676f",
            "text.color": "#25343b",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    blue = "#58788a"
    blue_dark = "#344f5b"
    blue_light = "#a8bdc7"
    grid_color = "#dfe7ea"
    generated: list[str] = []

    def save_figure(fig: Any, filename: str) -> None:
        destination = output_figures / filename
        if destination.exists() and not overwrite:
            fail(f"Output already exists: {destination}")
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        generated.append(filename)

    plot_columns = selected[:6]
    cols = 2
    rows = max(1, math.ceil(len(plot_columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows), squeeze=False)
    for axis, column in zip(axes.flat, plot_columns):
        if column in numeric_series:
            values = numeric_series[column].dropna()
            bins = min(15, max(6, int(math.sqrt(max(len(values), 1)))))
            axis.hist(values, bins=bins, color=blue, edgecolor="white", linewidth=0.8)
            axis.set_xlabel(f"{display_name(task, column)}（{display_unit(task, column) or '数值'}）")
        else:
            counts = dataframe[column].dropna().astype(str).value_counts().head(10)
            levels = category_levels_for(column, dataframe[column])[:10]
            counts = counts.reindex(levels).dropna()
            axis.bar(range(len(counts)), counts.values, color=blue_light, edgecolor=blue_dark, linewidth=0.7)
            axis.set_xticks(range(len(counts)), counts.index, rotation=25, ha="right")
            axis.set_xlabel(display_name(task, column))
        axis.set_title(f"{display_name(task, column)}分布")
        axis.set_ylabel("记录数")
        axis.grid(axis="y", color=grid_color, linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(plot_columns):]:
        axis.set_visible(False)
    fig.suptitle("主要变量的分布", fontsize=18, fontweight="bold", color=blue_dark, y=1.01)
    fig.tight_layout()
    save_figure(fig, "eda-distributions.png")

    outcome = str(spec.get("outcome"))
    factors = [str(value) for value in [*spec.get("predictors", []), *spec.get("controls", [])] if str(value) in dataframe.columns]
    relationship_columns = factors[:6]
    if relationship_columns:
        rows = max(1, math.ceil(len(relationship_columns) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4.4 * rows), squeeze=False)
        outcome_numeric = numeric_series.get(outcome)
        outcome_is_categorical = outcome in categorical_columns
        outcome_text = dataframe[outcome].dropna().astype(str)
        outcome_categories = list(dict.fromkeys(outcome_text.tolist()))
        if variable_roles.get(outcome) == "ordinal" and isinstance(
            spec.get("outcome_categories"), list
        ):
            approved_order = [str(value) for value in spec["outcome_categories"]]
            outcome_categories = [
                value for value in approved_order if value in set(outcome_categories)
            ]
        category_colors = [
            "#344f5b",
            "#58788a",
            "#7f9cab",
            "#a8bdc7",
            "#c7d6dc",
            "#77966d",
            "#c18c5d",
            "#8b6f91",
        ]

        def set_proportion_axis(axis: Any, label: str) -> None:
            ticks = np.linspace(0, 1, 6)
            axis.set_ylim(0, 1)
            axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
            axis.set_ylabel(label)

        for axis, factor in zip(axes.flat, relationship_columns):
            grid_axis = "y"
            factor_numeric = numeric_series.get(factor)
            if outcome_is_categorical and factor_numeric is not None:
                pair = pd.DataFrame(
                    {"x": factor_numeric, "outcome": dataframe[outcome].astype("string")}
                ).dropna()
                unique_x = int(pair["x"].nunique())
                bin_count = min(8, unique_x)
                if bin_count >= 2:
                    pair["group"] = pd.qcut(pair["x"], q=bin_count, duplicates="drop")
                    proportions = pd.crosstab(
                        pair["group"], pair["outcome"], normalize="index"
                    )
                    x_positions = np.arange(len(proportions.index))
                    plotted_categories = outcome_categories
                    if variable_roles.get(outcome) == "binary":
                        positive = str(
                            spec.get("positive_class")
                            if spec.get("positive_class") is not None
                            else outcome_categories[-1]
                        )
                        plotted_categories = [positive]
                    for index, category in enumerate(plotted_categories):
                        values = (
                            proportions[category]
                            if category in proportions.columns
                            else pd.Series(0.0, index=proportions.index)
                        )
                        axis.plot(
                            x_positions,
                            values,
                            marker="o",
                            linewidth=2,
                            color=category_colors[index % len(category_colors)],
                            label=str(category),
                        )
                    axis.set_xticks(
                        x_positions,
                        [str(value) for value in proportions.index],
                        rotation=25,
                        ha="right",
                    )
                    set_proportion_axis(
                        axis,
                        "正类比例"
                        if variable_roles.get(outcome) == "binary"
                        else "类别比例",
                    )
                    axis.legend(title=display_name(task, outcome), frameon=False)
                axis.set_xlabel(f"{display_name(task, factor)}分组")
            elif outcome_is_categorical:
                grouped = dataframe[[factor, outcome]].dropna().copy()
                grouped[factor] = grouped[factor].astype(str)
                grouped[outcome] = grouped[outcome].astype(str)
                factor_levels = category_levels_for(factor, grouped[factor])[:10]
                grouped = grouped[grouped[factor].isin(factor_levels)]
                proportions = pd.crosstab(
                    grouped[factor], grouped[outcome], normalize="index"
                ).reindex(factor_levels)
                y_positions = np.arange(len(factor_levels))
                left = np.zeros(len(factor_levels))
                for index, category in enumerate(outcome_categories):
                    values = (
                        proportions[category].fillna(0).to_numpy()
                        if category in proportions.columns
                        else np.zeros(len(factor_levels))
                    )
                    axis.barh(
                        y_positions,
                        values,
                        left=left,
                        color=category_colors[index % len(category_colors)],
                        label=str(category),
                    )
                    left += values
                axis.set_yticks(y_positions, factor_levels)
                ticks = np.linspace(0, 1, 6)
                axis.set_xlim(0, 1)
                axis.set_xticks(ticks, [f"{value:.0%}" for value in ticks])
                axis.set_xlabel("组内类别比例")
                axis.set_ylabel(display_name(task, factor))
                axis.legend(title=display_name(task, outcome), frameon=False)
                grid_axis = "x"
            elif outcome_numeric is not None and factor_numeric is not None:
                pair = pd.DataFrame({"x": factor_numeric, "y": outcome_numeric}).dropna()
                axis.scatter(pair["x"], pair["y"], s=28, color=blue, alpha=0.62, edgecolors="none")
                if len(pair) >= 3 and pair["x"].nunique() > 1:
                    slope, intercept = np.polyfit(pair["x"], pair["y"], 1)
                    x_line = np.linspace(pair["x"].min(), pair["x"].max(), 100)
                    axis.plot(x_line, slope * x_line + intercept, color=blue_dark, linewidth=2.2)
                axis.set_xlabel(f"{display_name(task, factor)}（{display_unit(task, factor) or '数值'}）")
                axis.set_ylabel(f"{display_name(task, outcome)}（{display_unit(task, outcome) or '数值'}）")
            elif outcome_numeric is not None:
                grouped = dataframe[[factor, outcome]].dropna()
                labels = [str(value) for value in grouped[factor].astype(str).value_counts().index[:10]]
                values = [pd.to_numeric(grouped.loc[grouped[factor].astype(str) == label, outcome], errors="coerce").dropna() for label in labels]
                values = [value for value in values if len(value)]
                labels = labels[: len(values)]
                if values:
                    boxes = axis.boxplot(values, tick_labels=labels, patch_artist=True)
                    for box in boxes["boxes"]:
                        box.set(facecolor=blue_light, edgecolor=blue_dark)
                axis.set_xlabel(display_name(task, factor))
                axis.set_ylabel(f"{display_name(task, outcome)}（{display_unit(task, outcome) or '数值'}）")
            axis.set_title(f"{display_name(task, factor)}与{display_name(task, outcome)}")
            axis.grid(axis=grid_axis, color=grid_color, linewidth=0.7, alpha=0.65)
            axis.spines[["top", "right"]].set_visible(False)
        for axis in list(axes.flat)[len(relationship_columns):]:
            axis.set_visible(False)
        fig.suptitle("主要变量之间的原始关系", fontsize=18, fontweight="bold", color=blue_dark, y=1.01)
        fig.tight_layout()
        save_figure(fig, "eda-relationships.png")

    numeric_columns = list(numeric_series)
    if len(numeric_columns) >= 2:
        correlation = pd.DataFrame({column: numeric_series[column] for column in numeric_columns}).corr()
        labels = [display_name(task, column) for column in numeric_columns]
        size = max(6.5, 1.25 * len(numeric_columns))
        fig, axis = plt.subplots(figsize=(size, size * 0.88))
        image = axis.imshow(correlation.values, cmap="Blues", vmin=-1, vmax=1)
        axis.set_xticks(range(len(labels)), labels, rotation=32, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                value = correlation.iloc[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 else blue_dark, fontsize=10)
        axis.set_title("数值变量相关矩阵", pad=18)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="相关系数")
        fig.tight_layout()
        save_figure(fig, "eda-correlations.png")

    table = (
        '<div class="table-wrap"><table><thead><tr><th>变量</th><th>类型</th><th class="num">有效记录</th><th class="num">缺失</th><th>分布摘要</th></tr></thead>'
        f'<tbody>{"".join(summary_rows)}</tbody></table></div>'
    )
    note = (
        '<p class="boundary-note">探索性图表展示未经模型调整的原始分布和两两关系，用于理解数据与发现异常；它们不能替代后续模型结果。</p>'
    )
    return table + build_figures_html(generated, output_figures) + note, generated


def build_method_html(
    spec: dict[str, Any], task: dict[str, Any], summary: dict[str, Any]
) -> str:
    model_type = spec.get("model_type")
    reasons = {
        "ols": "因变量按连续数值处理，使用 OLS 估计在控制其他变量后的条件均值差异。",
        "logistic": "因变量包含两个类别，使用二元 Logistic 回归估计结局优势比。",
        "poisson": "因变量为非负整数计数，使用 Poisson 回归估计发生率比。",
        "negative-binomial": "因变量为非负整数计数且波动明显超过均值，使用负二项回归容纳额外离散。",
        "multinomial-logistic": "因变量包含三个及以上无自然顺序的类别，使用多分类 Logistic 回归分别与已确认的参照类别比较。",
        "ordinal-logistic": "因变量包含三个及以上有自然顺序的类别，使用有序 Logistic 回归估计进入更高等级的累计优势。",
    }
    predictors = [display_name(task, value) for value in spec.get("predictors", [])]
    controls = [display_name(task, value) for value in spec.get("controls", [])]
    categorical = [display_name(task, value) for value in spec.get("categorical_columns", [])]
    positive = spec.get("positive_class")
    positive_html = f'<li><strong>正类定义：</strong><code>{esc(positive)}</code></li>' if positive is not None else ""
    outcome_reference = spec.get("outcome_reference_class")
    outcome_reference_html = (
        f'<li><strong>二分类结果基准类别：</strong><code>{esc(outcome_reference)}</code></li>'
        if outcome_reference is not None
        else ""
    )
    reference = spec.get("reference_class")
    reference_html = (
        f'<li><strong>结果参照类别：</strong><code>{esc(reference)}</code></li>'
        if reference is not None
        else ""
    )
    category_order = spec.get("outcome_categories")
    order_html = (
        f'<li><strong>结果类别顺序：</strong>{esc(" < ".join(map(str, category_order)))}</li>'
        if model_type == "ordinal-logistic" and isinstance(category_order, list)
        else ""
    )
    predictor_references = spec.get("categorical_reference_categories", {})
    predictor_reference_html = ""
    if isinstance(predictor_references, dict) and predictor_references:
        predictor_reference_items = "".join(
            "<li>"
            f"<strong>{esc(display_name(task, column))}：</strong>"
            f"<code>{esc(reference_value)}</code>"
            "</li>"
            for column, reference_value in sorted(predictor_references.items())
        )
        predictor_reference_html = (
            '<li><strong>分类自变量参照组：</strong><ul class="info-list nested-list">'
            f"{predictor_reference_items}</ul></li>"
        )
    continuous_forms = spec.get("continuous_functional_forms", {})
    continuous_form_html = ""
    if isinstance(continuous_forms, dict) and continuous_forms:
        form_items = "".join(
            "<li>"
            f"<strong>{esc(display_name(task, column))}：</strong>"
            f"{esc(item.get('plain_label', item.get('form', '—')) if isinstance(item, dict) else '—')}"
            + (
                "；结点（原单位）="
                + "、".join(fmt_number(value) for value in item.get("knots_original_units", []))
                if isinstance(item, dict) and item.get("form") == "restricted-cubic-spline"
                else ""
            )
            + (
                f"；来源={esc(CONTINUOUS_FORM_SOURCE_LABELS.get(str(item.get('selection_source')), item.get('selection_source', '未记录')))}；理由={esc(item.get('selection_rationale', '未记录'))}"
                if isinstance(item, dict)
                else ""
            )
            + "</li>"
            for column, item in sorted(continuous_forms.items())
        )
        continuous_form_html = (
            '<li><strong>连续变量函数形式：</strong><ul class="info-list nested-list">'
            f"{form_items}</ul></li>"
        )
    outcome = display_name(task, spec.get("outcome"))
    all_factors = [*predictors, *controls]
    factor_expression = " + ".join(
        f"β{index}×{factor}" for index, factor in enumerate(all_factors, 1)
    ) or "无解释变量"
    formulas = {
        "ols": f"{outcome} = β0 + {factor_expression} + 误差",
        "logistic": f"logit[P({outcome}=正类)] = β0 + {factor_expression}",
        "poisson": f"log[E({outcome})] = β0 + {factor_expression}",
        "negative-binomial": f"log[E({outcome})] = β0 + {factor_expression}，并估计额外离散参数 α",
        "multinomial-logistic": f"log[P({outcome}=类别 j) / P({outcome}=参照类别)] = β0j + {factor_expression}",
        "ordinal-logistic": f"logit[P({outcome}≤等级 j)] = 阈值j − ({factor_expression})",
    }
    assumptions = {
        "ols": [
            "结果变量按连续数值处理，主要关系可用线性形式近似；",
            "观测之间相互独立；",
            "残差方差基本稳定，残差分布没有严重偏离；",
            "解释因素之间不存在严重多重共线性，单个观测不会过度支配结果。",
        ],
        "logistic": [
            "结果变量只有两个类别且正类定义明确；",
            "观测之间相互独立，连续因素与对数优势之间关系基本合适；",
            "样本中有足够的正类和负类事件，不存在完全分离；",
            "解释因素之间不存在严重多重共线性或高影响观测。",
        ],
        "poisson": [
            "结果变量是非负整数计数；",
            "观测之间相互独立，均值结构采用对数链接；",
            f"调整后过度离散检查状态为{spec.get('count_dispersion_check', {}).get('status', '未记录')}，单侧p值为{fmt_number(spec.get('count_dispersion_check', {}).get('p_value'))}；未拒绝只表示当前样本未发现明确证据；",
            f"过多零值检查状态为{spec.get('zero_inflation_check', {}).get('status', '未记录')}；实际零值{spec.get('zero_inflation_check', {}).get('observed_zero_count', '—')}个，模型预计{fmt_number(spec.get('zero_inflation_check', {}).get('expected_zero_count'))}个；",
            "不存在已知的结构性过多零值、严重多重共线性或少数观测过度支配结果。",
        ],
        "negative-binomial": [
            "结果变量是非负整数计数，且其条件方差可以大于条件均值；",
            "观测之间相互独立，均值结构采用对数链接；",
            "额外离散由负二项参数合理描述，不存在无法解释的过多零值；",
            f"负二项必要性检查状态为{spec.get('negative_binomial_need_check', {}).get('status', '未记录')}，过多零值检查状态为{spec.get('zero_inflation_check', {}).get('status', '未记录')}；",
            "不存在严重多重共线性或少数观测过度支配结果。",
        ],
        "multinomial-logistic": [
            "结果变量含三个及以上互斥、无自然顺序的类别，参照类别已确认；",
            "观测之间相互独立，各类别有足够样本，不存在完全分离；",
            "连续因素与各类别相对参照类别的对数相对风险关系基本合适；",
            f"IIA敏感性检查状态为{spec.get('iia_check', {}).get('status', '未记录')}，最小Holm校正p值为{fmt_number(spec.get('iia_check', {}).get('minimum_adjusted_p_value'))}；未发现敏感性不等于证明IIA成立；",
            "解释因素之间不存在严重多重共线性。",
        ],
        "ordinal-logistic": [
            "结果类别有明确且经用户确认的自然顺序；",
            "观测之间相互独立，各等级有足够样本；",
            f"比例优势检查状态为{spec.get('proportional_odds_check', {}).get('status', '未记录')}，p值为{fmt_number(spec.get('proportional_odds_check', {}).get('p_value'))}；未拒绝只表示当前样本未发现明显违背证据；",
            "解释因素之间不存在严重多重共线性。",
        ],
    }
    return (
        '<div class="callout">'
        f'<strong>为什么选择这个模型：</strong>本次使用 {esc(MODEL_LABELS.get(str(model_type), model_type))}。'
        f'{esc(reasons.get(str(model_type), "模型由已批准的分析规格确定。"))}</div>'
        '<h3>模型表达</h3>'
        f'<pre class="code-block">{esc(formulas.get(str(model_type), "模型表达由已批准规格确定"))}</pre>'
        '<div class="two-column">'
        '<div class="panel"><h3>模型规格</h3><ul class="info-list">'
        f'<li><strong>用户希望回答：</strong>{esc(DECISION_GOAL_LABELS.get(str(task.get("decision_goal")), task.get("decision_goal") or "未单独记录"))}</li>'
        f'<li><strong>分析目标：</strong>{esc(GOAL_LABELS.get(str(spec.get("goal")), spec.get("goal")))}</li>'
        f'<li><strong>希望解释的结果：</strong>{esc(display_name(task, spec.get("outcome")))}</li>'
        f'<li><strong>重点关注的因素：</strong>{esc("、".join(predictors))}</li>'
        f'<li><strong>同时考虑的因素：</strong>{esc("、".join(controls) or "无")}</li>'
        f'<li><strong>分类变量：</strong><code>{esc(", ".join(map(str, categorical)) or "无")}</code></li>'
        f'{positive_html}{outcome_reference_html}{reference_html}{order_html}'
        f'{predictor_reference_html}{continuous_form_html}</ul></div>'
        '<div class="panel"><h3>不确定性设置</h3><ul class="info-list">'
        f'<li><strong>置信水平：</strong>{float(spec.get("confidence_level", .95)):.0%}</li>'
        f'<li><strong>标准误：</strong><code>{esc(spec.get("robust_se", "—"))}</code></li>'
        f'<li><strong>完整案例数：</strong>{esc(spec.get("complete_case_rows", "—"))}</li>'
        f'<li><strong>估计参数数：</strong>{esc(spec.get("estimated_parameter_count", "—"))}</li>'
        f'<li><strong>类别支持度复核：</strong>{esc(spec.get("category_support_screen", {}).get("status", "not-applicable"))}</li>'
        f'<li><strong>IIA敏感性复核：</strong>{esc(spec.get("iia_check", {}).get("status", "not-applicable"))}</li>'
        f'<li><strong>预测验证：</strong>{esc(summary.get("predictive_validation", {}).get("status", "not-applicable"))}</li>'
        f'<li><strong>过多零值复核：</strong>{esc(spec.get("zero_inflation_check", {}).get("status", "not-applicable"))}</li>'
        f'<li><strong>残差自由度：</strong>{esc(spec.get("estimated_residual_df", "—"))}</li>'
        '</ul></div></div>'
        '<h3>需要检查的模型条件</h3>'
        f'{html_list(esc(item) for item in assumptions.get(str(model_type), ["检查模型设定、样本独立性和估计稳定性。"]))}'
    )


def build_results_html(
    results: list[dict[str, str]],
    model_type: str,
    spec: dict[str, Any],
    task: dict[str, Any],
    compact: bool = False,
) -> str:
    rows: list[str] = []
    reference_map = spec.get("categorical_reference_categories", {})
    if not isinstance(reference_map, dict):
        reference_map = {}
    show_predictor_reference = bool(reference_map)
    for row in results:
        if row.get("term_type") not in {None, "", "coefficient"}:
            continue
        estimate = to_float(row.get("estimate"))
        direction = "正向" if estimate is not None and estimate > 0 else ("负向" if estimate is not None and estimate < 0 else "接近零")
        ratio_cell = ""
        if model_type in {
            "logistic",
            "poisson",
            "negative-binomial",
            "multinomial-logistic",
            "ordinal-logistic",
        }:
            ratio_cell = f'<td class="num">{fmt_number(row.get("exp_estimate"))}</td>'
        category_cell = (
            f'<td>{esc(row.get("outcome_category"))}</td>'
            if model_type == "multinomial-logistic"
            else ""
        )
        column, level = split_result_term(str(row.get("term") or ""), spec)
        predictor_reference_cell = (
            f'<td>{esc(reference_map.get(column) if level is not None else "—")}</td>'
            if show_predictor_reference
            else ""
        )
        factor_omnibus_cell = (
            f'<td class="num">{esc(fmt_p(row.get("factor_omnibus_p_value_adjusted_bh")))}</td>'
            if show_predictor_reference
            else ""
        )
        rows.append(
            "<tr>"
            f'{category_cell}'
            f'<td class="term">{esc(row.get("term"))}</td>'
            f'{predictor_reference_cell}'
            f'<td>{direction}</td>'
            f'<td class="num">{fmt_number(row.get("estimate"))}</td>'
            f'<td class="num">{fmt_number(row.get("std_error"))}</td>'
            f'<td class="num">[{fmt_number(row.get("ci_low"))}, {fmt_number(row.get("ci_high"))}]</td>'
            f'<td class="num">{esc(fmt_p(row.get("p_value")))}</td>'
            f'<td class="num">{esc(fmt_p(row.get("p_value_adjusted_bh")))}</td>'
            f'{factor_omnibus_cell}'
            f'<td class="num">{fmt_number(row.get("vif"))}</td>'
            f'<td>{"可作审慎解释" if interpretation_supported(row) else "限制单项解释"}</td>'
            f'{ratio_cell}'
            "</tr>"
        )
    ratio_header = ""
    if model_type == "logistic":
        ratio_header = '<th class="num">优势比（OR）</th>'
    elif model_type in {"poisson", "negative-binomial"}:
        ratio_header = '<th class="num">发生率比（IRR）</th>'
    elif model_type == "multinomial-logistic":
        ratio_header = '<th class="num">相对风险比（RRR）</th>'
    elif model_type == "ordinal-logistic":
        ratio_header = '<th class="num">累计优势比（OR）</th>'
    category_header = "<th>结果类别</th>" if model_type == "multinomial-logistic" else ""
    predictor_reference_header = "<th>自变量参照组</th>" if show_predictor_reference else ""
    factor_omnibus_header = (
        '<th class="num">变量总体BH校正p值</th>' if show_predictor_reference else ""
    )
    wrapper_class = "table-wrap compact-results" if compact else "table-wrap"
    return (
        f'<div class="{wrapper_class}"><table><thead><tr>{category_header}<th>模型项</th>{predictor_reference_header}<th>方向</th><th class="num">估计值</th><th class="num">标准误</th><th class="num">置信区间</th><th class="num">原始p值</th><th class="num">BH校正p值</th>{factor_omnibus_header}<th class="num">VIF</th><th>解释状态</th>'
        f'{ratio_header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def build_statistical_quiz_html(
    tested_rows: list[dict[str, str]],
    model_type: str,
    spec: dict[str, Any],
    summary: dict[str, Any],
    task: dict[str, Any],
    alpha: float,
) -> str:
    if not tested_rows:
        return ""
    row = tested_rows[0]
    column, level = split_result_term(str(row.get("term") or ""), spec)
    factor = display_name(task, column)
    if level is not None:
        factor += f"（{level}）"
    if row.get("outcome_category"):
        factor += f"对结果类别“{row.get('outcome_category')}”"
    p_value = to_float(row.get("p_value"))
    low = to_float(row.get("ci_low"))
    high = to_float(row.get("ci_high"))
    interval_clear = (
        low is not None and high is not None and not (low <= 0 <= high)
    )
    adjusted_p = to_float(row.get("p_value_adjusted_bh"))
    evidence_clear = interpretation_supported(row) and interval_clear
    collinearity_restricted = to_bool(row.get("collinearity_restricted")) is True
    if collinearity_restricted:
        correct_one = (
            f"{factor}的单项结果受到严重共线性限制；即使原始或校正p值较小，"
            "也不能据此稳定拆分它的独立作用，应先修订变量组合或做敏感性分析。"
        )
    elif evidence_clear:
        correct_one = (
            f"现有样本支持{factor}与结果存在条件关联，但仍需结合效应大小、"
            "置信区间、诊断和研究设计，不能直接表述为因果。"
        )
    else:
        correct_one = (
            f"现有样本不足以排除{factor}与结果没有独立关联的可能；"
            "这不等于已经证明二者完全无关。"
        )

    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    bp = to_float(metrics.get("breusch_pagan_p_value"))
    dispersion = to_float(metrics.get("dispersion"))
    if bp is not None:
        diagnostic_fact = f"本次 Breusch–Pagan 检验 p={fmt_number(bp)}"
        diagnostic_answer = (
            "未见明显异方差证据，但仍要结合残差图与样本量。"
            if bp >= alpha
            else "存在异方差风险，应优先采用稳健标准误并做敏感性检查。"
        )
    elif dispersion is not None:
        diagnostic_fact = f"本次计数模型离散度={fmt_number(dispersion)}"
        diagnostic_answer = (
            "波动明显超过 Poisson 预期，需要考虑负二项等替代模型。"
            if dispersion > 1.5
            else "未见明显过度离散，但仍需检查零值和高影响观测。"
        )
    else:
        accuracy = to_float(metrics.get("classification_accuracy"))
        diagnostic_fact = (
            f"本次样本内分类准确率={fmt_number(accuracy)}"
            if accuracy is not None
            else "本次分类模型已给出系数区间和影响诊断"
        )
        diagnostic_answer = (
            "不能只凭样本内准确率判断可靠性，还要检查类别不平衡、区间宽度和新数据验证。"
        )

    return (
        '<div class="learning-lab">'
        '<h3>统计检验小课堂：用本次结果做判断</h3>'
        '<p class="lead">先选出你认为最合理的解释，再展开答案。题目使用的就是本次分析结果。</p>'
        '<article class="quiz-card">'
        f'<p class="quiz-question">1. {esc(factor)}这一项的原始p值为 {esc(fmt_p(p_value))}，'
        f'BH校正p值为 {esc(fmt_p(adjusted_p))}，'
        f'{float(spec.get("confidence_level", .95)):.0%} 置信区间为 '
        f'[{esc(fmt_number(low))}, {esc(fmt_number(high))}]。哪种解释更恰当？</p>'
        '<ol class="quiz-options"><li>只要 p 值小于 0.05，就证明存在因果作用。</li>'
        f'<li>{esc(correct_one)}</li>'
        '<li>只看 p 值即可判断实际影响是否足够大。</li></ol>'
        f'<details><summary>查看答案与解释</summary><p><strong>答案：B。</strong>{esc(correct_one)} '
        'p 值描述数据与无关联假设的不一致程度；效应大小、区间和设计边界仍不可省略。</p></details>'
        '</article>'
        '<article class="quiz-card">'
        f'<p class="quiz-question">2. {esc(diagnostic_fact)}。据此，下一步最合理的判断是什么？</p>'
        '<ol class="quiz-options"><li>诊断指标只用于装饰报告，可以忽略。</li>'
        f'<li>{esc(diagnostic_answer)}</li>'
        '<li>只要某个系数显著，任何诊断问题都不会影响结论。</li></ol>'
        f'<details><summary>查看答案与解释</summary><p><strong>答案：B。</strong>{esc(diagnostic_answer)}</p></details>'
        '</article></div>'
    )


def build_factor_omnibus_html(
    summary: dict[str, Any],
    task: dict[str, Any],
    alpha: float,
    limit: int | None = MAIN_DETAIL_LIMIT,
    compact: bool = False,
) -> str:
    multiplicity = summary.get("multiplicity", {})
    tests = (
        multiplicity.get("categorical_omnibus_tests", [])
        if isinstance(multiplicity, dict)
        else []
    )
    if not isinstance(tests, list) or not tests:
        return '<p class="detail-reference">本次模型没有需要总体检验的分类自变量。</p>'
    collinearity = summary.get("collinearity", {})
    severe_factors = set(
        map(str, collinearity.get("severe_factors", []))
        if isinstance(collinearity, dict)
        else []
    )
    displayed_tests = tests if limit is None else tests[:limit]
    rows: list[str] = []
    for item in displayed_tests:
        if not isinstance(item, dict):
            continue
        adjusted = to_float(item.get("p_value_adjusted_bh"))
        factor_name = str(item.get("factor"))
        if factor_name in severe_factors:
            decision = "受严重共线性限制，先修订或做敏感性分析"
        elif adjusted is not None and adjusted < alpha:
            decision = "可继续查看具体类别"
        else:
            decision = "不展开具体类别强结论"
        rows.append(
            "<tr>"
            f'<td>{esc(display_name(task, item.get("factor")))}</td>'
            f'<td>{esc(item.get("reference_category"))}</td>'
            f'<td class="num">{esc(item.get("degrees_of_freedom"))}</td>'
            f'<td class="num">{esc(fmt_p(item.get("p_value")))}</td>'
            f'<td class="num">{esc(fmt_p(adjusted))}</td>'
            f'<td>{esc(decision)}</td>'
            "</tr>"
        )
    wrapper_class = "table-wrap compact-results" if compact else "table-wrap"
    remainder_note = (
        f'<p class="detail-reference">正文仅展示前 {limit} 项；完整总体检验见技术与诊断明细。</p>'
        if limit is not None and len(tests) > limit
        else ""
    )
    return (
        f'<div class="{wrapper_class}"><table><thead><tr><th>分类变量</th><th>参照组</th>'
        '<th class="num">自由度</th><th class="num">原始p值</th>'
        '<th class="num">BH校正p值</th><th>后续解释</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>{remainder_note}'
    )


def build_continuous_shape_html(
    summary: dict[str, Any], task: dict[str, Any], alpha: float
) -> str:
    tests = summary.get("continuous_shape_tests", [])
    if not isinstance(tests, list) or not tests:
        return '<p class="detail-reference">本次模型没有需要函数形式检验的连续变量。</p>'
    labels = {
        "linear": "线性",
        "quadratic": "线性加二次项",
        "restricted-cubic-spline": "4结点限制性立方样条",
    }
    rows: list[str] = []
    for item in tests:
        if not isinstance(item, dict):
            continue
        nonlinear_p = to_float(item.get("nonlinear_p_value_adjusted_bh"))
        if item.get("form") == "linear":
            conclusion = "本规格按线性关系估计；没有额外非线性项可检验"
        elif nonlinear_p is not None and nonlinear_p < alpha:
            conclusion = "非线性成分有证据；优先结合调整后预测曲线解释整体形状"
        else:
            conclusion = "未见明确非线性证据；不能据此证明关系严格为直线"
        rows.append(
            "<tr>"
            f'<td>{esc(display_name(task, item.get("variable")))}</td>'
            f'<td>{esc(labels.get(str(item.get("form")), item.get("form")))}</td>'
            f'<td class="num">{esc(fmt_p(item.get("overall_p_value")))}</td>'
            f'<td class="num">{esc(fmt_p(item.get("nonlinear_p_value")))}</td>'
            f'<td class="num">{esc(fmt_p(nonlinear_p))}</td>'
            f'<td>{esc(conclusion)}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>连续变量</th><th>已批准形式</th>'
        '<th class="num">整体关系p值</th><th class="num">非线性p值</th>'
        '<th class="num">非线性BH校正p值</th><th>如何理解</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="boundary-note">二次项和样条基函数共同表示一条曲线，不能把单个基函数系数解释为现实中的独立效应。曲线形状以调整后预测图为主。</p>'
    )


def build_statistical_tests_html(
    results: list[dict[str, str]], model_type: str, spec: dict[str, Any], summary: dict[str, Any], task: dict[str, Any]
) -> str:
    confidence_level = float(spec.get("confidence_level", 0.95))
    alpha = 1.0 - confidence_level
    tested_rows = [
        row
        for row in results
        if row.get("term") != "const"
        and row.get("term_type") in {None, "", "coefficient"}
        and to_bool(row.get("nonlinear_basis_term")) is not True
        and row.get("continuous_functional_form") not in {
            "quadratic", "restricted-cubic-spline"
        }
    ]
    significant_rows = [row for row in tested_rows if interpretation_supported(row)]
    interval_rows = [
        row for row in tested_rows
        if to_float(row.get("ci_low")) is not None
        and to_float(row.get("ci_high")) is not None
        and (to_float(row.get("ci_low")) > 0 or to_float(row.get("ci_high")) < 0)
    ]
    coefficient_test = "t 检验" if model_type == "ols" else "Wald z 检验"
    ratio_notes = {
        "ols": "线性回归系数的无关联参考值为 0。",
        "logistic": "二元 Logistic 模型同时报告优势比（OR），无关联参考值为 1。",
        "poisson": "Poisson 模型同时报告发生率比（IRR），无关联参考值为 1。",
        "negative-binomial": "负二项模型同时报告发生率比（IRR），无关联参考值为 1。",
        "multinomial-logistic": "多分类模型报告各类别相对参照类别的相对风险比（RRR），无关联参考值为 1。",
        "ordinal-logistic": "有序模型报告进入更高等级的累计优势比（OR），无关联参考值为 1。",
    }
    ratio_note = ratio_notes.get(model_type, "模型系数的无关联参考值为 0。")
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    overall_items: list[str] = []
    overall_p = to_float(metrics.get("f_p_value"))
    overall_test_name = "整体 F 检验"
    if overall_p is None:
        overall_p = to_float(metrics.get("llr_p_value"))
        overall_test_name = "整体似然比检验"
    if overall_p is not None:
        overall_items.append(
            f"{overall_test_name}：{fmt_p_statement(overall_p)}。它比较当前模型与仅含截距的基准模型。"
        )
    if not overall_items:
        overall_items.append("当前运行摘要没有记录单独的整体显著性检验 p 值，参数检验应逐项结合区间解释。")
    overall_items.append(
        f"在显著性水平 α={alpha:.2f} 下，{len(significant_rows)}/{len(tested_rows)} 个非截距项同时通过多重比较且未受严重共线性限制；"
        f"{len(interval_rows)}/{len(tested_rows)} 个常规区间未跨越无关联参考值。"
    )
    intuitive_items: list[str] = []
    if overall_p is not None:
        if overall_p < alpha:
            intuitive_items.append(
                "把这些因素放在一起后，模型比“完全不考虑这些因素”的基准模型更能解释样本中的结果差异；"
                f"整体检验为 {fmt_p_statement(overall_p)}。这说明至少有一个纳入因素与结果有关，但不能据此断言存在因果作用。"
            )
        else:
            intuitive_items.append(
                "整体检验没有提供足够证据表明当前这组因素合起来优于仅含截距的基准模型，"
                "因此不宜对单个系数作强结论。"
            )
    clear_labels: list[str] = []
    uncertain_labels: list[str] = []
    for row in tested_rows:
        term = str(row.get("term") or "")
        column, level = split_result_term(term, spec)
        factor = display_name(task, column)
        if level is not None:
            reference_map = spec.get("categorical_reference_categories", {})
            reference = str(
                reference_map.get(column)
                if isinstance(reference_map, dict) and reference_map.get(column) is not None
                else variable_metadata(task).get(column, {}).get("reference_category") or "参照组"
            )
            factor = f"{factor}（{level} 相对 {reference}）"
        if row.get("outcome_category"):
            factor += (
                f"｜结果为“{row.get('outcome_category')}”相对"
                f"“{row.get('reference_class') or spec.get('reference_class')}”"
            )
        low = to_float(row.get("ci_low"))
        high = to_float(row.get("ci_high"))
        if interpretation_supported(row):
            clear_labels.append(factor)
        else:
            uncertain_labels.append(factor)
    if clear_labels:
        shown = "、".join(clear_labels[:MAIN_SUMMARY_LIMIT])
        suffix = f"等 {len(clear_labels)} 项" if len(clear_labels) > MAIN_SUMMARY_LIMIT else ""
        intuitive_items.append(
            f"同时通过总体检验（如适用）、BH校正且未受严重共线性限制的项目共 {len(clear_labels)} 项，主要包括{shown}{suffix}；"
            "方向与效应大小仍需结合模型参数和研究设计解释。"
        )
    if uncertain_labels:
        shown = "、".join(uncertain_labels[:MAIN_SUMMARY_LIMIT])
        suffix = f"等 {len(uncertain_labels)} 项" if len(uncertain_labels) > MAIN_SUMMARY_LIMIT else ""
        intuitive_items.append(
            f"另有 {len(uncertain_labels)} 项未通过完整证据门槛或受到严重共线性限制，"
            f"包括{shown}{suffix}；现有数据不足以判断稳定方向。"
        )
    quiz_html = build_statistical_quiz_html(
        tested_rows, model_type, spec, summary, task, alpha
    )
    return (
        '<div class="callout"><strong>这部分检验回答什么？</strong>'
        '统计检验用于判断当前样本中的关联是否可能只是随机波动造成的。它不回答影响是否足够重要，也不能把关联证明为因果。</div>'
        '<div class="two-column">'
        '<div class="panel"><h3>检验设置</h3><ul class="info-list">'
        f'<li><strong>参数检验：</strong>{coefficient_test}</li>'
        f'<li><strong>置信水平：</strong>{confidence_level:.0%}</li>'
        f'<li><strong>显著性水平：</strong>α={alpha:.2f}</li>'
        f'<li><strong>参考值：</strong>{esc(ratio_note)}</li>'
        '</ul></div>'
        '<div class="panel"><h3>总体检验概览</h3>'
        f'{html_list(esc(item) for item in overall_items)}</div></div>'
        '<h3>分类变量总体检验</h3>'
        f'{build_factor_omnibus_html(summary, task, alpha)}'
        '<h3>连续变量函数形式与非线性检验</h3>'
        f'{build_continuous_shape_html(summary, task, alpha)}'
        '<h3>本次数据的检验结论</h3>'
        f'{html_list(esc(item) for item in intuitive_items)}'
        '<h3>参数检验结果</h3>'
        '<p class="detail-reference">完整参数表已移至“技术与诊断明细”，正文不再逐项重复同一解释句式。</p>'
        '<p class="boundary-note">统计显著性表示数据与无关联假设之间的不一致程度，不等于实际影响很大、结论必然正确或存在因果关系。应同时查看效应大小、置信区间、模型诊断和研究设计。本报告保留原始p值用于审计，并以 Benjamini–Hochberg 校正结果及分类变量总体检验控制多重比较造成的偶然发现。</p>'
        f'{quiz_html}'
    )


def figure_data_uri(figures_dir: Path, name: str) -> str:
    figure_path = (figures_dir / name).resolve()
    try:
        figure_path.relative_to(figures_dir.resolve())
    except ValueError:
        fail(f"Figure path escapes the output figures directory: {name}")
    if not figure_path.is_file():
        fail(f"Figure referenced by the report does not exist: {figure_path}")
    mime_type = mimetypes.guess_type(figure_path.name)[0]
    if mime_type not in {"image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"}:
        fail(f"Unsupported report figure type: {figure_path.name}")
    encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_figure_block(figures_dir: Path, name: str, title: str, caption: str) -> str:
    source = figure_data_uri(figures_dir, name)
    return (
        '<figure>'
        f'<button class="figure-open" type="button" data-figure-title="{esc(title)}" '
        f'data-source-file="figures/{esc(name)}" aria-label="放大查看：{esc(title)}">'
        f'<img src="{source}" alt="{esc(title)}" loading="eager" decoding="sync"></button>'
        f'<figcaption><strong>{esc(title)}</strong>{esc(caption)}</figcaption>'
        '</figure>'
    )


def build_figures_html(figure_names: list[str], figures_dir: Path) -> str:
    blocks: list[str] = []
    for name in figure_names:
        title, caption = FIGURE_METADATA.get(
            name,
            (Path(name).stem.replace("-", " "), "该图由已批准的统计分析流程生成。"),
        )
        blocks.append(build_figure_block(figures_dir, name, title, caption))
    return f'<div class="figure-grid">{"".join(blocks)}</div>' if blocks else '<p class="lead">本次分析没有生成图表。</p>'


def build_model_figures_html(
    figure_names: list[str],
    figures_dir: Path,
    summary: dict[str, Any],
    diagnostics: list[dict[str, str]],
    results: list[dict[str, str]],
    spec: dict[str, Any],
    task: dict[str, Any],
) -> str:
    model_type = str(summary.get("model_type") or spec.get("model_type"))
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    confidence_level = float(spec.get("confidence_level", 0.95))
    tested_rows = [row for row in results if row.get("term") != "const"]
    clear_rows = [
        row for row in tested_rows
        if to_float(row.get("ci_low")) is not None
        and to_float(row.get("ci_high")) is not None
        and (to_float(row.get("ci_low")) > 0 or to_float(row.get("ci_high")) < 0)
    ]
    bp = to_float(metrics.get("breusch_pagan_p_value"))
    jb = to_float(metrics.get("jarque_bera_p_value"))
    rmse = to_float(metrics.get("rmse"))
    outcome_unit = display_unit(task, spec.get("outcome"))
    influential = summary.get("influential_cleaned_data_rows", [])
    influence_summary = summary.get("influence_diagnostics", {})
    captions: dict[str, str] = {
        "coefficients.png": (
            f"本次模型的 {len(tested_rows)} 个非截距项中，有 {len(clear_rows)} 个 {confidence_level:.0%} 置信区间未跨过 0。"
            "区间未跨 0 表明在当前模型设定下，关联方向较明确；具体影响大小见“结果发现”。"
        ),
        "residuals-vs-fitted.png": (
            (
                f"本次 Breusch–Pagan 检验 p={fmt_number(bp)}，未发现残差波动随拟合值系统变化的强证据；"
                if bp is not None and bp >= .05
                else (
                    f"本次 Breusch–Pagan 检验 p={fmt_number(bp)}，提示残差波动可能随拟合值变化；"
                    if bp is not None
                    else "本次运行未记录 Breusch–Pagan 检验结果；"
                )
            )
            + (
                f"模型拟合值与实际值的典型偏差约为 {fmt_number(rmse)}{outcome_unit or '个结果单位'}。"
                if rmse is not None
                else "仍需结合图中点云是否围绕零线分布来判断模型形式。"
            )
        ),
        "residual-qq.png": (
            (
                f"本次 Jarque–Bera 检验 p={fmt_number(jb)}，未达到 0.05 显著性水平，"
                "没有发现残差明显偏离正态分布的强证据；图中尾部仍有少量点偏离参考线，结论应保持适度谨慎。"
                if jb is not None and jb >= .05
                else (
                    f"本次 Jarque–Bera 检验 p={fmt_number(jb)}，残差分布与正态形态存在明显差异；"
                    "置信区间和显著性结论需要谨慎解释。"
                    if jb is not None
                    else "本次运行未记录残差正态性检验结果，应直接检查点是否系统偏离参考线。"
                )
            )
        ),
        "influence.png": (
            f"本次 {esc(influence_summary.get('method', '模型特定影响筛查'))} 标记了 {len(influential)} 条候选记录；"
            f"对优先级最高的 {esc(influence_summary.get('case_deletion_evaluated_count', 0))} 条进行了逐条删一重拟合。"
            "原模型始终保留，候选标记不能作为自动删除依据；完整结果见末尾“技术与诊断明细”。"
        ),
    }
    blocks: list[str] = []
    for name in figure_names:
        title, generic_caption = FIGURE_METADATA.get(
            name,
            (Path(name).stem.replace("-", " "), "该图由已批准的统计分析流程生成。"),
        )
        if name.startswith("adjusted-category-probabilities-"):
            title = (
                "连续变量与各类别概率的调整后关系"
                if model_type == "multinomial-logistic"
                else "连续变量与各等级概率的调整后关系"
            )
            generic_caption = (
                f"每条曲线表示在其他模型变量固定于代表值或参照组时，一个结果"
                f"{'类别' if model_type == 'multinomial-logistic' else '等级'}的预测概率，"
                f"同色阴影为{confidence_level:.0%}置信带。"
                "类别编码没有被当作连续数值；曲线用于整体解释获批的二次项或限制性立方样条，不单独解释基函数系数。"
            )
        elif name.startswith("adjusted-probability-"):
            title = "连续变量与结果概率的调整后关系"
            generic_caption = (
                f"曲线显示在其他模型变量固定于代表值或参照组时的预测正类概率，阴影为{confidence_level:.0%}置信带。"
                "它用于解释获批的二次项或限制性立方样条整体形状，不把任何单个基函数系数解释为现实效应。"
            )
        elif name.startswith("adjusted-outcome-"):
            title = "连续变量与结果的调整后关系"
            generic_caption = (
                f"曲线显示在其他模型变量固定于代表值或参照组时的预测结果，阴影为{confidence_level:.0%}置信带。"
                "它用于解释获批的二次项或限制性立方样条整体形状，不把任何单个基函数系数解释为现实效应。"
            )
        elif name.startswith("adjusted-count-"):
            title = "连续变量与预计计数的调整后关系"
            generic_caption = (
                f"曲线显示在其他模型变量固定于代表值或参照组时的预计计数，阴影为{confidence_level:.0%}置信带。"
                "它用于解释预先确定的限制性立方样条整体形状，不把任何单个基函数系数解释为现实效应。"
            )
        caption = captions.get(name, generic_caption)
        blocks.append(build_figure_block(figures_dir, name, title, caption))
    return f'<div class="figure-grid">{"".join(blocks)}</div>' if blocks else '<p class="lead">本次分析没有生成诊断图。</p>'


def validate_standalone_report(
    report_html: str, expected_figure_names: list[str], figures_dir: Path
) -> None:
    required_layout_fragments = {
        'class="hero"': "dark report hero",
        'class="page-shell"': "two-column report shell",
        'class="toc"': "left table of contents",
        "position: sticky": "sticky table of contents",
        'class="print-button screen-only"': "print/PDF button",
        'id="print-report"': "print/PDF button hook",
        "prepareImagesForPrint": "image-aware print preparation",
        "window.print()": "browser print action",
        'class="figure-grid"': "bounded figure grid",
        "max-height: var(--figure-max-height)": "scaled figure preview",
    }
    missing_layout = [
        label for fragment, label in required_layout_fragments.items() if fragment not in report_html
    ]
    if missing_layout:
        fail(f"Report template is missing required layout features: {missing_layout}")

    image_sources = re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', report_html, flags=re.IGNORECASE)
    if len(image_sources) != len(expected_figure_names):
        fail(
            "Standalone report figure count mismatch: "
            f"expected {len(expected_figure_names)}, found {len(image_sources)}"
        )
    non_embedded = [source for source in image_sources if not source.startswith("data:image/")]
    if non_embedded:
        fail(f"Standalone report still contains non-embedded images: {non_embedded[:3]}")
    source_files = re.findall(r'\bdata-source-file=["\']([^"\']+)["\']', report_html, flags=re.IGNORECASE)
    expected_sources = [f"figures/{name}" for name in expected_figure_names]
    if source_files != expected_sources:
        fail("Standalone report embedded-figure inventory does not match generated figures")
    for name, source in zip(expected_figure_names, image_sources):
        try:
            encoded = source.split(",", 1)[1]
            embedded_bytes = base64.b64decode(encoded, validate=True)
        except (IndexError, ValueError) as exc:
            fail(f"Cannot decode embedded figure {name}: {exc}")
        if embedded_bytes != (figures_dir / name).read_bytes():
            fail(f"Embedded figure bytes do not match outputs/figures/{name}")
    if re.search(r'<(?:link|script)\b[^>]*(?:href|src)=["\']https?://', report_html, flags=re.IGNORECASE):
        fail("Standalone report contains an external CSS or JavaScript dependency")


def build_diagnostics_html(
    summary: dict[str, Any], diagnostics: list[dict[str, str]], spec: dict[str, Any], task: dict[str, Any]
) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    model_type = str(summary.get("model_type") or spec.get("model_type"))
    applicability = summary.get("diagnostic_applicability", {})
    if not isinstance(applicability, dict):
        applicability = {}
    rows: list[tuple[str, str, str]] = []
    predictive_validation = summary.get("predictive_validation", {})
    if predictive_validation.get("status") == "completed":
        validation_metrics = predictive_validation.get("metrics", {})
        compact_metrics = "；".join(
            f"{METRIC_LABELS.get(str(key), str(key))}={fmt_number(value)}"
            for key, value in validation_metrics.items()
            if value is not None
        )
        rows.append((
            "模型对未参与该折拟合的数据表现如何",
            compact_metrics or "已完成折外验证",
            f"采用{predictive_validation.get('method')}。这些是内部折外预测结果，优先于样本内准确率或RMSE，但仍不能替代独立外部数据验证。",
        ))
    elif str(spec.get("goal")) == "prediction":
        rows.append((
            "新数据预测表现是否经过内部验证",
            "未能评估",
            f"原因：{predictive_validation.get('reason', '未记录')}。报告不得把样本内拟合指标写成新数据预测能力。",
        ))
    r_squared = to_float(metrics.get("r_squared"))
    adjusted_r_squared = to_float(metrics.get("adjusted_r_squared"))
    if r_squared is not None:
        adjusted_text = (
            f"，考虑模型包含的因素数量后为 {adjusted_r_squared:.1%}"
            if adjusted_r_squared is not None
            else ""
        )
        rows.append((
            "模型解释了多少样本差异",
            f"R²={r_squared:.3f}",
            f"模型解释了当前样本中约 {r_squared:.1%} 的结果差异{adjusted_text}。"
            "这是样本内拟合程度，不代表因果解释率，也不等同于新数据上的预测准确率。",
        ))
    rmse = to_float(metrics.get("rmse"))
    if rmse is not None:
        outcome = display_name(task, spec.get("outcome"))
        unit = display_unit(task, spec.get("outcome"))
        rows.append((
            "拟合值通常与实际值差多少",
            f"RMSE={fmt_number(rmse)}",
            f"模型给出的{outcome}拟合值与实际值之间，典型偏差约为 {fmt_number(rmse)}{unit or '个单位'}；"
            "数值越小越好，但是否足够小需要结合实际使用场景判断。",
        ))
    dw = to_float(metrics.get("durbin_watson"))
    if dw is not None:
        rows.append((
            "相邻残差是否同向聚集",
            f"Durbin–Watson={fmt_number(dw)}",
            (
                "该值接近 2，未见明显的一阶残差相关信号。"
                if 1.5 <= dw <= 2.5
                else "该值明显偏离 2，残差可能存在顺序相关，需要结合数据采集顺序进一步检查。"
            ),
        ))
    bp = to_float(metrics.get("breusch_pagan_p_value"))
    jb = to_float(metrics.get("jarque_bera_p_value"))
    dispersion = to_float(metrics.get("dispersion"))
    negative_binomial_alpha = to_float(metrics.get("negative_binomial_alpha"))
    classification_accuracy = to_float(metrics.get("classification_accuracy"))
    roc_auc = to_float(metrics.get("roc_auc"))
    brier_score = to_float(metrics.get("brier_score"))
    calibration_error = to_float(metrics.get("calibration_mean_absolute_error"))
    multiclass_log_loss = to_float(metrics.get("multiclass_log_loss"))
    ordinal_error = to_float(metrics.get("ordinal_mean_absolute_category_error"))
    count_rmse = to_float(metrics.get("count_rmse"))
    vif_values = [
        to_float(row.get("value"))
        for row in diagnostics
        if str(row.get("metric", "")).startswith("VIF:")
    ]
    vif_values = [value for value in vif_values if value is not None]
    collinearity = summary.get("collinearity", {})
    if not isinstance(collinearity, dict):
        collinearity = {}
    collinearity_status = str(collinearity.get("status", ""))
    has_structured_collinearity = collinearity_status in {"clear", "review", "severe"}
    severe_vif_terms = collinearity.get("severe_terms", [])
    if not isinstance(severe_vif_terms, list):
        severe_vif_terms = []
    moderate_vif_terms = collinearity.get("moderate_terms", [])
    if not isinstance(moderate_vif_terms, list):
        moderate_vif_terms = []
    structural_basis_high_vif_terms = collinearity.get(
        "joint_shape_terms_excluded_from_individual_vif_gate", []
    )
    if not isinstance(structural_basis_high_vif_terms, list):
        structural_basis_high_vif_terms = []
    maximum_vif = to_float(collinearity.get("maximum_vif"))
    if maximum_vif is None and vif_values:
        maximum_vif = max(vif_values)
    if bp is not None:
        rows.append((
            "误差波动是否随拟合水平改变",
            f"Breusch–Pagan p={fmt_number(bp)}",
            (
                "p<0.05，残差波动可能不稳定，普通标准误可能不够可靠，应结合残差图并考虑稳健标准误。"
                if bp < .05
                else "p≥0.05，本次数据没有显示残差波动系统变化的强证据；仍应结合残差图检查局部结构。"
            ),
        ))
    if jb is not None:
        rows.append((
            "残差分布是否明显偏离正态",
            f"Jarque–Bera p={fmt_number(jb)}",
            (
                "p<0.05，残差分布与正态形态存在明显差异，置信区间和显著性结论需谨慎。"
                if jb < .05
                else "p≥0.05，本次数据没有提供残差明显偏离正态分布的强证据；Q-Q 图尾部仍需单独检查。"
            ),
        ))
    if maximum_vif is not None:
        if has_structured_collinearity and collinearity_status == "severe":
            vif_explanation = (
                f"普通模型项中有 {len(severe_vif_terms)} 项达到 VIF≥10，存在严重共线性；"
                "受影响的单项系数已限制为不可作稳定独立作用解释。"
            )
        elif has_structured_collinearity and collinearity_status == "review":
            vif_explanation = (
                f"普通模型项中有 {len(moderate_vif_terms)} 项的 VIF 介于 5 与 10 之间，"
                "需要关注系数不稳定风险，但尚未达到严重门槛。"
            )
        elif has_structured_collinearity and structural_basis_high_vif_terms:
            vif_explanation = (
                "最高 VIF 来自需要整体解释的非线性基函数；这些基函数不逐项解释，"
                "因此没有触发普通模型项的严重共线性限制。"
            )
        elif has_structured_collinearity:
            vif_explanation = "普通模型项未达到需要关注的多重共线性门槛。"
        else:
            vif_explanation = (
                "VIF≥10，存在严重共线性；相关单项系数已限制为不可作稳定独立作用解释。"
                if maximum_vif >= 10
                else (
                    "VIF 介于 5 与 10 之间，需要关注系数不稳定风险。"
                    if maximum_vif >= 5
                    else "所有 VIF 均低于 5，未见明显的多重共线性问题。"
                )
            )
        rows.append((
            "纳入因素是否彼此高度重叠",
            f"最大 VIF={fmt_number(maximum_vif)}",
            vif_explanation,
        ))
    if dispersion is not None:
        rows.append((
            "计数结果的实际波动是否超过模型预期",
            f"离散度={fmt_number(dispersion)}",
            "离散度明显高于 1，存在过度离散迹象，应考虑替代计数模型。"
            if dispersion > 1.5
            else "离散度接近 1，未见明显过度离散迹象。",
        ))
    if model_type in {"poisson", "negative-binomial"} and count_rmse is not None:
        rows.append((
            "拟合计数通常与实际计数相差多少",
            f"计数 RMSE={fmt_number(count_rmse)}",
            "该值需要结合结果计数的实际范围理解，并配合实际计数—拟合计数图检查系统偏差。",
        ))
    if negative_binomial_alpha is not None:
        rows.append((
            "计数波动是否需要超出 Poisson 的额外离散",
            f"负二项 α={fmt_number(negative_binomial_alpha)}",
            (
                "α 明显大于 0，负二项模型正在容纳超过 Poisson 均值—方差关系的额外波动。"
                if negative_binomial_alpha > .1
                else "α 接近 0，额外离散较弱；可把 Poisson 作为经确认的敏感性比较，而不能据此自动换模。"
            ),
        ))
    if model_type == "negative-binomial":
        need = summary.get("negative_binomial_need_check", {})
        rows.append((
            "是否确实需要负二项的额外离散参数",
            f"必要性检查={need.get('status', '未记录')}；p={fmt_number(need.get('p_value'))}",
            "只有调整后波动显著超过Poisson范围时才支持负二项；这项判断不会触发自动换模。",
        ))
    if model_type in {"poisson", "negative-binomial"}:
        zero_check = summary.get("zero_inflation_check", {})
        rows.append((
            "普通计数模型能否解释零值数量",
            f"实际零值={zero_check.get('observed_zero_count', '—')}；预计零值={fmt_number(zero_check.get('expected_zero_count'))}；p={fmt_number(zero_check.get('p_value'))}",
            "未检出过多零值不等于证明没有结构性零值；检出时需要专门的门槛或零膨胀工作流，不能自动换模。",
        ))
    if classification_accuracy is not None:
        rows.append((
            "模型在当前样本中分对了多少类别",
            f"样本内准确率={classification_accuracy:.1%}",
            "这是同一批建模数据上的分类表现，不是新数据准确率；还需结合类别比例、混淆情况和外部验证。",
        ))
    if model_type == "logistic" and roc_auc is not None:
        rows.append((
            "模型能否区分正类与负类",
            f"ROC AUC={fmt_number(roc_auc)}",
            "AUC只描述排序区分能力；概率是否可信还必须结合校准图和Brier分数。",
        ))
        rows.append((
            "预测概率是否与实际比例一致",
            f"Brier={fmt_number(brier_score)}；校准误差={fmt_number(calibration_error)}",
            "数值越小通常越好，但应结合校准图、正类比例和具体使用场景判断。",
        ))
    if model_type == "multinomial-logistic" and multiclass_log_loss is not None:
        rows.append((
            "多分类概率预测是否稳定",
            f"对数损失={fmt_number(multiclass_log_loss)}",
            "需结合混淆矩阵和各类别比例校准；总体准确率可能掩盖小类别错分。",
        ))
        iia = summary.get("iia_check", {})
        rows.append((
            "移除一个结果类别后，其余类别的相对关系是否稳定",
            f"IIA检查={iia.get('status', '未记录')}；最小Holm校正p={fmt_number(iia.get('minimum_adjusted_p_value'))}",
            "该检查比较逐一删除非参照类别前后的共同系数；未发现敏感性不等于证明IIA成立，还要结合类别是否天然相似或嵌套判断。",
        ))
    if model_type == "ordinal-logistic" and ordinal_error is not None:
        rows.append((
            "预测等级平均偏离多少级",
            f"平均绝对等级误差={fmt_number(ordinal_error)}",
            "同时检查有序混淆矩阵和等级比例校准；该指标不能替代比例优势假设检查。",
        ))
        proportional_odds = summary.get("proportional_odds_check", {})
        rows.append((
            "各等级分界能否共用一组影响系数",
            f"比例优势检查={proportional_odds.get('status', '未记录')}；p={fmt_number(proportional_odds.get('p_value'))}",
            "当前模型只在该检查未发现明显违背证据时才能运行；未拒绝不等于假设已被证明，样本量和类别稀疏仍会影响检查能力。",
        ))
    for label, key in (
        ("普通残差正态 Q-Q 诊断", "normal_qq"),
        ("线性回归异方差诊断", "heteroskedasticity"),
    ):
        if applicability.get(key) == "not-applicable":
            rows.append((label, "不适用", f"该诊断不属于 {model_type} 模型的成立条件，因此未生成对应图形或通过性结论。"))
    influence_count = len(summary.get("influential_cleaned_data_rows", []))
    influence_summary = summary.get("influence_diagnostics", {})
    influence_available = influence_summary.get("status") == "available"
    if influence_available:
        evaluated = int(influence_summary.get("case_deletion_evaluated_count", 0) or 0)
        maximum_change = influence_summary.get("maximum_standardized_parameter_change")
        flip_count = int(influence_summary.get("sign_flip_count", 0) or 0)
        significance_flip_count = int(influence_summary.get("significance_flip_count", 0) or 0)
        rows.append((
            "是否有少数记录明显影响结果",
            f"模型特定候选={influence_count} 条；删一重拟合={evaluated} 条",
            f"最大标准化参数变化={fmt_number(maximum_change)}，系数方向改变={flip_count}项，显著性判断改变={significance_flip_count}项。"
            "候选记录没有被自动删除；完整明细见末尾附录。",
        ))
    else:
        rows.append((
            "是否有少数记录明显影响结果",
            "未能评估",
            f"原因：{influence_summary.get('reason', '未记录')}。不能把“未计算”解释为“没有高影响记录”。",
        ))
    table_rows = "".join(
        "<tr>"
        f"<td><strong>{esc(question)}</strong></td>"
        f'<td class="num">{esc(value)}</td>'
        f"<td>{esc(explanation)}</td>"
        "</tr>"
        for question, value, explanation in rows
    )
    collinearity_concern = (
        collinearity_status in {"review", "severe"}
        if has_structured_collinearity
        else maximum_vif is not None and maximum_vif >= 5
    )
    concerns = sum(
        [
            1 if bp is not None and bp < .05 else 0,
            1 if jb is not None and jb < .05 else 0,
            1 if collinearity_concern else 0,
            1 if dispersion is not None and dispersion > 1.5 else 0,
        ]
    )
    overall = {
        "ols": (
            "OLS的残差形态、方差稳定性和共线性诊断未显示已记录的明显问题；"
            if concerns == 0
            else f"OLS诊断中有 {concerns} 项需要进一步关注；"
        ),
        "logistic": "Logistic诊断应联合解释ROC区分能力、概率校准、分离检查和影响观测；",
        "poisson": "Poisson诊断应重点解释离散度、计数残差和实际/拟合计数偏差；",
        "negative-binomial": "负二项诊断应重点解释离散参数、计数残差和实际/拟合计数偏差；",
        "multinomial-logistic": "多分类诊断应联合解释混淆矩阵、类别比例校准、对数损失和稀疏类别；",
        "ordinal-logistic": "有序分类诊断应联合解释有序混淆矩阵、等级误差、等级比例校准，并单独核查比例优势假设；",
    }.get(model_type, "请按模型类型解释诊断结果；")
    severe_collinearity = (
        collinearity_status == "severe"
        if has_structured_collinearity
        else maximum_vif is not None and maximum_vif >= 10
    )
    if severe_collinearity:
        overall += "检测到严重共线性，受影响单项系数只能保留审计，必须先修订变量组合或做经批准的敏感性分析。"
    if not influence_available:
        overall += "当前模型的影响诊断未能完成，不能据此声称没有高影响记录。"
    elif influence_count:
        overall += (
            f"发现 {influence_count} 条候选高影响记录；已保留原模型并完成"
            f" {influence_summary.get('case_deletion_completed_count', 0)} 次逐条删一比较。"
        )
    else:
        overall += "模型特定筛查未发现达到候选参考线的高影响记录。"
    return (
        f'<div class="callout"><strong>综合判断：</strong>{esc(overall)}</div>'
        '<div class="table-wrap"><table><thead><tr><th>诊断关注点</th><th class="num">本次结果</th><th>如何理解本次结果</th></tr></thead>'
        f'<tbody>{table_rows}</tbody></table></div>'
    )


def build_diagnostic_appendix_html(
    profile: dict[str, Any],
    prep: dict[str, Any],
    summary: dict[str, Any],
    diagnostics: list[dict[str, str]],
    results: list[dict[str, str]],
    spec: dict[str, Any],
    task: dict[str, Any],
    influence_rows: list[dict[str, str]] | None = None,
) -> str:
    influence_rows = influence_rows or []
    _sheet, columns = profile_summary(profile)
    missing_columns = [
        column for column in columns if int(column.get("missing_count") or 0) > 0
    ]
    missing_columns.sort(
        key=lambda column: float(column.get("missing_rate") or 0), reverse=True
    )
    sensitive, unresolved_sensitive = sensitive_field_review(profile, task)
    missing_rows = "".join(
        "<tr>"
        f'<td class="term">{esc(column.get("column"))}</td>'
        f'<td class="num">{esc(column.get("missing_count", 0))}</td>'
        f'<td class="num">{float(column.get("missing_rate") or 0):.1%}</td>'
        "</tr>"
        for column in missing_columns
    )
    action_rows = "".join(
        "<tr>"
        f'<td class="term">{esc(action.get("decision_id"))}</td>'
        f'<td>{esc(CHOICE_LABELS.get(str(action.get("selected_option")), action.get("selected_option")))}</td>'
        f'<td>{esc((action.get("details") or {}).get("column", "—") if isinstance(action.get("details"), dict) else "—")}</td>'
        f'<td class="num">{esc(action.get("affected_row_count", 0))}</td>'
        f'<td class="num">{esc(action.get("rows_after", "—"))}</td>'
        "</tr>"
        for action in prep.get("actions", [])
        if isinstance(action, dict)
    )
    deferred_items = [
        esc(item.get("reason"))
        for item in prep.get("deferred_actions", [])
        if isinstance(item, dict)
    ]
    metric_rows = "".join(
        "<tr>"
        f'<td>{esc(row.get("category"))}</td>'
        f'<td class="term">{esc(METRIC_LABELS.get(str(row.get("metric")), row.get("metric")))}</td>'
        f'<td class="num">{fmt_number(row.get("value"))}</td>'
        f'<td>{esc(METRIC_DESCRIPTIONS.get(str(row.get("metric")), "该指标用于保留分析运行的完整审计记录。"))}</td>'
        "</tr>"
        for row in diagnostics
    )
    influence_summary = summary.get("influence_diagnostics", {})
    influence_table_rows = "".join(
        "<tr>"
        f'<td class="num">{esc(row.get("cleaned_data_row"))}</td>'
        f'<td>{esc(row.get("candidate_reasons"))}</td>'
        f'<td class="num">{fmt_number(row.get("leverage"))}</td>'
        f'<td class="num">{fmt_number(row.get("standardized_residual"))}</td>'
        f'<td class="num">{fmt_number(row.get("cook_distance"))}</td>'
        f'<td>{esc(row.get("refit_status"))}</td>'
        f'<td class="num">{fmt_number(row.get("max_standardized_parameter_change"))}</td>'
        f'<td class="num">{esc(row.get("sign_flip_count") if row.get("sign_flip_count") not in (None, "") else "—")}</td>'
        f'<td class="num">{esc(row.get("significance_flip_count") if row.get("significance_flip_count") not in (None, "") else "—")}</td>'
        "</tr>"
        for row in influence_rows
    ) or '<tr><td colspan="9">没有候选记录，或该项未能评估；请以状态说明为准。</td></tr>'
    influence_note = (
        f'<p><strong>评估状态：</strong>{esc(influence_summary.get("status", "未记录"))}；'
        f'<strong>方法：</strong>{esc(influence_summary.get("method", "未记录"))}；'
        f'<strong>删一敏感性：</strong>{esc(influence_summary.get("sensitivity_status", "未评估"))}。'
        '候选记录没有被自动删除。</p>'
        '<div class="table-wrap compact-results"><table><thead><tr><th>清洗数据行</th><th>候选原因</th><th>杠杆值</th><th>标准化残差</th><th>Cook距离</th><th>删一重拟合</th><th>最大标准化参数变化</th><th>方向改变数</th><th>显著性改变数</th></tr></thead>'
        f'<tbody>{influence_table_rows}</tbody></table></div>'
    )
    missing_body = missing_rows or '<tr><td colspan="3">未检测到缺失字段。</td></tr>'
    action_body = action_rows or '<tr><td colspan="5">无处理动作记录。</td></tr>'
    missingness_impact_rows = "".join(
        "<tr>"
        f'<td>{esc(row.get("column"))}</td>'
        f'<td>{esc(row.get("role"))}</td>'
        f'<td class="num">{esc(row.get("missing_rows", 0))}</td>'
        f'<td class="num">{esc(row.get("observed_rows", 0))}</td>'
        f'<td class="num">{fmt_number(row.get("max_outcome_proportion_gap"))}</td>'
        f'<td class="num">{fmt_number(row.get("outcome_mean_difference"))}</td>'
        "</tr>"
        for row in prep.get("missingness_bias_rows", [])
        if isinstance(row, dict)
    ) or '<tr><td colspan="6">所选字段没有处理前缺失比较项。</td></tr>'
    category_support = spec.get("category_support_screen", {})
    support_categories = [
        str(value) for value in category_support.get("outcome_categories", [])
    ] if isinstance(category_support, dict) else []
    support_rows = flatten_category_support_rows(category_support) if isinstance(category_support, dict) else []
    support_header = "".join(f'<th class="num">{esc(category)}数</th>' for category in support_categories)
    support_body = "".join(
        "<tr>"
        f'<td>{esc(row.get("factor"))}</td>'
        f'<td>{esc(row.get("level"))}</td>'
        f'<td class="num">{esc(row.get("total"))}</td>'
        + "".join(
            f'<td class="num">{esc(row.get(f"outcome_count:{category}", 0))}</td>'
            for category in support_categories
        )
        + f'<td>{esc(row.get("risk_codes") or "无")}</td>'
        + "</tr>"
        for row in support_rows
    ) or '<tr><td colspan="4">本次模型没有分类变量支持度明细。</td></tr>'
    return (
        '<p class="lead">正文只保留会影响理解和决策的摘要；以下小字号明细框保留完整处理与诊断记录，供复核和审计。</p>'
        '<div class="detail-box">'
        '<h3>完整模型参数（紧凑表）</h3>'
        '<p>完整结果集中保留在此处，避免在正文逐项铺开；屏幕阅读时可在表格内滚动查看。</p>'
        f'{build_results_html(results, str(spec.get("model_type")), spec, task, compact=True)}'
        '</div>'
        '<div class="detail-box">'
        '<h3>完整分类变量总体检验</h3>'
        f'{build_factor_omnibus_html(summary, task, 1.0 - float(spec.get("confidence_level", 0.95)), limit=None, compact=True)}'
        '</div>'
        '<div class="detail-box">'
        '<h3>每个类别的结果支持度（紧凑表）</h3>'
        '<p>这些计数用于检查稀疏类别与分类分离；当前获批模型只能包含已通过复核的类别集合。</p>'
        '<div class="table-wrap compact-results"><table><thead><tr><th>变量</th><th>类别</th><th class="num">总数</th>'
        f'{support_header}<th>风险标记</th></tr></thead><tbody>{support_body}</tbody></table></div>'
        '</div>'
        '<div class="detail-box">'
        '<h3>缺失字段完整清单</h3>'
        '<p>下表来自原始数据体检；它不会因处理后数据已无缺失而被改写。</p>'
        '<div class="table-wrap"><table><thead><tr><th>字段</th><th class="num">缺失数</th><th class="num">缺失率</th></tr></thead>'
        f'<tbody>{missing_body}</tbody></table></div>'
        '<h4>所选字段缺失组与非缺失组比较</h4>'
        '<p>比例差按结果类别占比的最大绝对差记录；数值结果则记录均值差。两者均为描述性指标，不能识别缺失机制。</p>'
        '<div class="table-wrap compact-results"><table><thead><tr><th>字段</th><th>角色</th><th class="num">缺失组数</th><th class="num">非缺失组数</th><th class="num">结果比例最大差</th><th class="num">结果均值差</th></tr></thead>'
        f'<tbody>{missingness_impact_rows}</tbody></table></div>'
        '<h4>潜在敏感字段完整清单</h4>'
        f'<p class="detail-sequence">{esc("、".join(sensitive) or "名称筛查未识别")}</p>'
        f'<p>尚待语义核对的无含义列名：{esc("、".join(unresolved_sensitive) or "无")}</p>'
        '</div>'
        '<div class="detail-box">'
        '<h3>数据处理与延后事项完整清单</h3>'
        '<div class="table-wrap"><table><thead><tr><th>决定</th><th>用户选择</th><th>字段</th><th class="num">影响记录数</th><th class="num">处理后记录数</th></tr></thead>'
        f'<tbody>{action_body}</tbody></table></div>'
        '<h4>延后事项</h4>'
        f'{html_list(deferred_items)}'
        '</div>'
        '<div class="detail-box">'
        '<h3>完整模型诊断数值</h3>'
        '<p>AIC、BIC 等比较型指标只有在同一数据上比较候选模型时才应重点解释。</p>'
        '<div class="table-wrap"><table><thead><tr><th>类别</th><th>指标</th><th class="num">数值</th><th>指标用途</th></tr></thead>'
        f'<tbody>{metric_rows}</tbody></table></div>'
        '<h4>模型特定高影响记录与逐条删一敏感性</h4>'
        f'{influence_note}'
        '</div>'
    )


def build_limitations_html(
    profile: dict[str, Any], spec: dict[str, Any], summary: dict[str, Any], task: dict[str, Any], prep: dict[str, Any] | None = None
) -> str:
    prep = prep or {}
    sensitive, unresolved_sensitive = sensitive_field_review(profile, task)
    items = [
        "本分析基于观测数据和已批准的统计模型，能够描述条件关联，但不能单独识别因果效应。",
        "未观测混杂、测量误差、样本选择和模型设定偏差仍可能影响估计结果。",
        "效应方向、大小、置信区间和诊断结果应共同解释；p 值不能作为唯一判断标准。",
        "模型结论只适用于当前样本、变量定义和处理规则，不能自动推广到其他总体或情境。",
        "高影响观测和异常值诊断用于敏感性检查，不能仅凭自动阈值删除记录。",
    ]
    missingness_screen = prep.get("missingness_bias_screen", {})
    missingness_contract = prep.get("missingness_conclusion_contract", {})
    if missingness_screen.get("status") == "review-required":
        items.append(
            "所选字段存在原始缺失。缺失组与非缺失组的样本构成比较只能提示选择风险，"
            "不能判定缺失机制；本流程没有完成不同缺失处理方法下的模型估计敏感性分析，"
            "因此相关结论只适用于实际进入模型的分析样本。"
        )
    if missingness_contract.get("model_estimate_sensitivity_completed") is not True:
        items.append("尚无模型级缺失值敏感性证据，不应声称结论对完整案例、单次填补或多重填补均稳健。")
    for warning in [*spec.get("warnings", []), *summary.get("warnings", [])]:
        text = str(warning)
        if text and text not in items:
            items.append(text)
    if sensitive:
        items.append(
            "处理后数据仍可能包含潜在敏感字段（"
            + "、".join(map(str, sensitive))
            + "），公开共享前需要单独进行隐私审查和必要的脱敏。"
        )
    if unresolved_sensitive:
        items.append(
            f"仍有 {len(unresolved_sensitive)} 个无含义列名尚未完成敏感性语义核对，"
            "不能把名称筛查未命中解释为不存在敏感字段。"
        )
    return html_list(esc(item) for item in items)


def purpose_for(path: str) -> str:
    name = Path(path).name
    purposes = {
        "final-report.html": "最终网页报告",
        "report-manifest.json": "文件清单与 SHA-256 校验值",
        "analysis.py": "可复现统计分析代码",
        "cleaned-data.csv": "处理后的分析数据",
        "model-results.csv": "模型系数、区间、原始p值与BH校正p值",
        "model-diagnostics.csv": "模型诊断指标",
        "factor-omnibus-tests.csv": "分类变量总体检验与多重比较校正结果",
        "category-support-screen.csv": "分类变量各类别与结果类别的支持度和分离风险记录",
        "continuous-shape-tests.csv": "连续变量总体关系与非线性成分联合检验",
        "influence-diagnostics.csv": "模型特定高影响记录筛查与逐条删一敏感性结果",
        "iia-check.json": "多分类Logistic无关替代独立性敏感性检查",
        "proportional-odds-check.json": "有序Logistic比例优势假设检查",
        "count-dispersion-check.json": "普通Poisson调整后过度离散门禁",
        "negative-binomial-need-check.json": "负二项额外离散参数必要性检查",
        "zero-inflation-check.json": "计数模型调整后过多零值检查",
        "predictive-validation.json": "预测任务确定性交叉验证结果",
        "model-summary.json": "模型摘要和解释边界",
        "analysis-run-log.json": "分析运行环境与产物日志",
        "data-profile.json": "原始数据体检结果",
        "approved-analysis-task.json": "经问卷回执批准的研究问题和变量角色",
        "report-approval.json": "最终报告生成审批回执",
        "data-preparation-log.json": "数据处理执行记录",
        "missingness-impact.csv": "缺失组与非缺失组的描述性样本构成比较",
        "approved-data-preparation-plan.json": "用户批准的数据处理方案",
        "approved-model-specification.json": "用户批准的模型规格",
    }
    return purposes.get(name, FIGURE_METADATA.get(name, ("分析图表", ""))[0])


def file_manifest_entry(
    path: Path, relative_path: str, include_size_text: bool = False
) -> dict[str, Any]:
    """Build stable metadata for a file included in the report bundle."""
    size = path.stat().st_size
    entry: dict[str, Any] = {
        "path": relative_path.replace("\\", "/"),
        "purpose": purpose_for(relative_path),
        "size": size,
    }
    if include_size_text:
        entry["size_text"] = fmt_size(size)
    entry["sha256"] = sha256_file(path)
    return entry


def build_reproducibility_html(deliverables: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f'<td><a class="file-link" href="{esc(item["path"])}">{esc(item["path"])}</a></td>'
        f'<td>{esc(item["purpose"])}</td>'
        f'<td class="num">{esc(item["size_text"])}</td>'
        "</tr>"
        for item in deliverables
    )
    command = (
        '& $env:INNO_DATA_ANALYSIS_PYTHON .\\analysis.py '
        '--data .\\cleaned-data.csv '
        '--spec .\\approved-model-specification.json '
        '--preparation-log .\\data-preparation-log.json '
        '--output-dir .\\reproduced-results'
    )
    return (
        '<p class="lead">报告目录保存了分析代码、处理后数据、模型规格、结果表、诊断记录和图表。'
        '运行前请确保工作区虚拟环境已安装报告清单中记录的统计依赖。</p>'
        '<h3>复现命令</h3>'
        f'<pre class="code-block">{esc(command)}</pre>'
        '<h3>生成文件</h3>'
        '<div class="table-wrap"><table><thead><tr><th>相对路径</th><th>用途</th><th class="num">大小</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
        '<p class="boundary-note">每个交付文件的 SHA-256 校验值记录在 '
        '<a href="report-manifest.json">report-manifest.json</a> 中。</p>'
    )


def main() -> None:
    args = parse_args()
    paths = {
        "profile": Path(args.profile).resolve(),
        "analysis_task": Path(args.analysis_task).resolve(),
        "preparation_log": Path(args.preparation_log).resolve(),
        "missingness_impact": Path(args.missingness_impact).resolve(),
        "model_spec": Path(args.model_spec).resolve(),
        "model_results": Path(args.model_results).resolve(),
        "model_diagnostics": Path(args.model_diagnostics).resolve(),
        "factor_tests": Path(args.factor_tests).resolve(),
        "category_support": Path(args.category_support).resolve(),
        "shape_tests": Path(args.shape_tests).resolve(),
        "influence_diagnostics": Path(args.influence_diagnostics).resolve(),
        "iia_check": Path(args.iia_check).resolve(),
        "proportional_odds_check": Path(args.proportional_odds_check).resolve(),
        "count_dispersion_check": Path(args.count_dispersion_check).resolve(),
        "negative_binomial_need_check": Path(args.negative_binomial_need_check).resolve(),
        "zero_inflation_check": Path(args.zero_inflation_check).resolve(),
        "predictive_validation": Path(args.predictive_validation).resolve(),
        "model_summary": Path(args.model_summary).resolve(),
        "analysis_run_log": Path(args.analysis_run_log).resolve(),
        "cleaned_data": Path(args.cleaned_data).resolve(),
        "analysis_code": Path(args.analysis_code).resolve(),
    }
    figures_dir = Path(args.figures_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    template_path = Path(__file__).resolve().parent.parent / "assets" / "report-template.html"
    if not figures_dir.is_dir():
        fail(f"Figures directory does not exist: {figures_dir}")
    if not template_path.is_file():
        fail(f"Report template does not exist: {template_path}")

    profile = read_json(paths["profile"], "data profile")
    task = read_json(paths["analysis_task"], "analysis task")
    prep = read_json(paths["preparation_log"], "preparation log")
    spec = read_json(paths["model_spec"], "model specification")
    summary = read_json(paths["model_summary"], "model summary")
    approval_path = Path(args.approval_record).resolve()
    try:
        report_approval = verify_approval(
            approval_path, "approve-final-report", paths["model_summary"]
        )
    except ApprovalError as exc:
        fail(str(exc))
    run_log = read_json(paths["analysis_run_log"], "analysis run log")
    results = read_csv_rows(paths["model_results"], "model results")
    diagnostics = read_csv_rows(paths["model_diagnostics"], "model diagnostics")
    factor_tests = read_csv_rows(paths["factor_tests"], "factor omnibus tests")
    category_support_rows = read_csv_rows(paths["category_support"], "category support screen")
    shape_tests = read_csv_rows(paths["shape_tests"], "continuous shape tests")
    influence_rows = read_csv_rows(paths["influence_diagnostics"], "influence diagnostics")
    iia = read_json(paths["iia_check"], "IIA check")
    proportional_odds = read_json(paths["proportional_odds_check"], "proportional-odds check")
    count_dispersion = read_json(paths["count_dispersion_check"], "count-dispersion check")
    negative_binomial_need = read_json(
        paths["negative_binomial_need_check"], "negative-binomial need check"
    )
    zero_inflation = read_json(paths["zero_inflation_check"], "excess-zero check")
    predictive_validation = read_json(
        paths["predictive_validation"], "predictive validation"
    )
    missingness_impact_rows = read_csv_rows(paths["missingness_impact"], "missingness impact")
    expected_missingness_rows = prep.get("missingness_bias_rows")
    if not isinstance(expected_missingness_rows, list) or len(missingness_impact_rows) != len(expected_missingness_rows):
        fail("Missingness-impact CSV differs from the preparation log")
    validate_artifacts(
        profile,
        task,
        prep,
        spec,
        summary,
        run_log,
        paths["cleaned_data"],
        paths["preparation_log"],
        paths["model_spec"],
        results,
        factor_tests,
        shape_tests,
        category_support_rows,
        influence_rows,
        iia,
        proportional_odds,
        count_dispersion,
        negative_binomial_need,
        zero_inflation,
        predictive_validation,
    )

    artifact_mapping = [
        (paths["profile"], "data-profile.json"),
        (paths["analysis_task"], "approved-analysis-task.json"),
        (paths["preparation_log"], "data-preparation-log.json"),
        (paths["missingness_impact"], "missingness-impact.csv"),
        (paths["model_spec"], "approved-model-specification.json"),
        (paths["model_results"], "model-results.csv"),
        (paths["model_diagnostics"], "model-diagnostics.csv"),
        (paths["factor_tests"], "factor-omnibus-tests.csv"),
        (paths["category_support"], "category-support-screen.csv"),
        (paths["shape_tests"], "continuous-shape-tests.csv"),
        (paths["influence_diagnostics"], "influence-diagnostics.csv"),
        (paths["iia_check"], "iia-check.json"),
        (paths["proportional_odds_check"], "proportional-odds-check.json"),
        (paths["count_dispersion_check"], "count-dispersion-check.json"),
        (paths["negative_binomial_need_check"], "negative-binomial-need-check.json"),
        (paths["zero_inflation_check"], "zero-inflation-check.json"),
        (paths["predictive_validation"], "predictive-validation.json"),
        (paths["model_summary"], "model-summary.json"),
        (paths["analysis_run_log"], "analysis-run-log.json"),
        (paths["cleaned_data"], "cleaned-data.csv"),
        (paths["analysis_code"], "analysis.py"),
        (approval_path, "report-approval.json"),
    ]
    if args.preparation_plan:
        artifact_mapping.append(
            (Path(args.preparation_plan).resolve(), "approved-data-preparation-plan.json")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_figures = output_dir / "figures"
    output_figures.mkdir(parents=True, exist_ok=True)
    final_report_path = output_dir / "final-report.html"
    manifest_path = output_dir / "report-manifest.json"
    if not args.overwrite and (final_report_path.exists() or manifest_path.exists()):
        fail("final-report.html or report-manifest.json already exists; use --overwrite only after approval")

    for source, relative in artifact_mapping:
        copy_artifact(source, output_dir / relative, args.overwrite)

    figure_names = [str(name) for name in summary.get("figures", [])]
    copied_figures: list[str] = []
    for name in figure_names:
        source = figures_dir / name
        destination = output_figures / name
        copy_artifact(source, destination, args.overwrite)
        copied_figures.append(name)

    eda_html, eda_figures = build_eda_artifacts(
        paths["cleaned_data"], task, spec, output_figures, args.overwrite
    )

    copied_paths = [relative for _source, relative in artifact_mapping] + [
        f"figures/{name}" for name in [*eda_figures, *copied_figures]
    ]
    deliverables: list[dict[str, Any]] = []
    for relative in copied_paths:
        path = output_dir / relative
        deliverables.append(file_manifest_entry(path, relative, include_size_text=True))
    deliverables.append(
        {
            "path": "final-report.html",
            "purpose": purpose_for("final-report.html"),
            "size": 0,
            "size_text": "当前文件",
            "sha256": None,
        }
    )
    deliverables.append(
        {
            "path": "report-manifest.json",
            "purpose": purpose_for("report-manifest.json"),
            "size": 0,
            "size_text": "生成后记录",
            "sha256": None,
        }
    )

    model_type = str(summary.get("model_type"))
    generated_at = datetime.now().astimezone()
    title = str(
        task.get("report_title")
        or spec.get("report_title")
        or f'{display_name(task, spec.get("outcome"))}相关因素分析报告'
    )
    replacements = {
        "{{REPORT_TITLE}}": esc(title),
        "{{DATASET_INTRO}}": esc(dataset_intro(profile, task)),
        "{{ROWS_USED}}": esc(summary.get("rows_used", "—")),
        "{{GENERATED_AT}}": esc(generated_at.strftime("%Y-%m-%d %H:%M %Z")),
        "{{CORE_CONCLUSION}}": esc(build_core_conclusion(spec, results, task, prep)),
        "{{PLAIN_FINDINGS_HTML}}": build_plain_findings_html(results, spec, task),
        "{{DATASET_HTML}}": build_dataset_html(profile, task, prep, spec, summary),
        "{{VARIABLES_HTML}}": build_variables_html(task),
        "{{QUALITY_HTML}}": build_quality_html(profile, prep, task),
        "{{EDA_HTML}}": eda_html,
        "{{METHOD_HTML}}": build_method_html(spec, task, summary),
        "{{STATISTICAL_TESTS_HTML}}": build_statistical_tests_html(
            results, model_type, spec, summary, task
        ),
        "{{MODEL_FIGURES_HTML}}": build_model_figures_html(
            copied_figures, output_figures, summary, diagnostics, results, spec, task
        ),
        "{{DIAGNOSTICS_HTML}}": build_diagnostics_html(summary, diagnostics, spec, task),
        "{{DIAGNOSTIC_APPENDIX_HTML}}": build_diagnostic_appendix_html(
            profile, prep, summary, diagnostics, results, spec, task, influence_rows
        ),
        "{{LIMITATIONS_HTML}}": build_limitations_html(profile, spec, summary, task, prep),
        "{{REPRODUCIBILITY_HTML}}": build_reproducibility_html(deliverables),
        "{{FOOTER_TEXT}}": esc(
            "本报告由已批准的数据处理与统计分析产物自动生成；报告本身未重新拟合模型。"
        ),
    }
    report_html = template_path.read_text(encoding="utf-8")
    for placeholder, replacement in replacements.items():
        report_html = report_html.replace(placeholder, replacement)
    unresolved = [token for token in replacements if token in report_html]
    if unresolved or "{{" in report_html or "}}" in report_html:
        fail("Report template contains unresolved placeholders")
    embedded_figure_names = [*eda_figures, *copied_figures]
    validate_standalone_report(report_html, embedded_figure_names, output_figures)
    atomic_write(final_report_path, report_html)

    report_entry = file_manifest_entry(final_report_path, "final-report.html")
    manifest_entries = [
        {key: value for key, value in item.items() if key != "size_text"}
        for item in deliverables
        if item["path"] not in {"final-report.html", "report-manifest.json"}
    ]
    manifest_entries.append(report_entry)
    manifest = {
        "status": "completed",
        "final_report_generated": True,
        "generated_at": generated_at.isoformat(),
        "report": "final-report.html",
        "report_mode": "standalone",
        "embedded_figures": len(embedded_figure_names),
        "model_type": model_type,
        "rows_used": summary.get("rows_used"),
        "approval": approval_summary(report_approval, approval_path),
        "files": sorted(manifest_entries, key=lambda item: item["path"]),
    }
    atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": True,
                "status": "completed",
                "final_report_generated": True,
                "report": str(final_report_path),
                "manifest": str(manifest_path),
                "copied_files": len(manifest_entries) - 1,
                "figures": len(eda_figures) + len(copied_figures),
                "eda_figures": len(eda_figures),
                "report_mode": "standalone",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
