#!/usr/bin/env python3
"""Build a draft, non-mutating data-preparation plan for an approved analysis task."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "execute-data-preparation" / "scripts"),
)
from missingness_bias import build_missingness_screen, flatten_missingness_screen  # noqa: E402
from file_utils import sha256_file as file_sha256  # noqa: E402
from execute_preparation import load_table  # noqa: E402


NUMERIC_TYPES = {"numeric-continuous", "numeric-discrete"}
CATEGORICAL_TYPES = {"categorical", "boolean"}
RISKY_TYPES = {"identifier-candidate", "sensitive-candidate", "all-missing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/data-preparation"))
    return parser.parse_args()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def choose_profile_sheet(profile: dict[str, Any], sheet_name: str) -> dict[str, Any]:
    for sheet in profile.get("profiles", []):
        if sheet.get("sheet") == sheet_name:
            return sheet
    available = [item.get("sheet") for item in profile.get("profiles", [])]
    raise ValueError(f"Task sheet {sheet_name!r} not found in profile. Available sheets: {available}")


def confirmed_table_read_spec(profile: dict[str, Any], sheet_name: str) -> dict[str, Any]:
    document = profile.get("table_read_spec")
    sheets = document.get("sheets") if isinstance(document, dict) else None
    if not isinstance(sheets, list):
        raise ValueError("Profile has no table-read specification; re-run profiling")
    item = next(
        (value for value in sheets if isinstance(value, dict) and value.get("sheet") == sheet_name),
        None,
    )
    if not isinstance(item, dict) or item.get("status") not in {"auto-confirmed", "user-confirmed"}:
        raise ValueError(f"Table structure for sheet {sheet_name!r} is not confirmed")
    selected = item.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"Table-read specification for sheet {sheet_name!r} is invalid")
    return selected


def pending(
    decision_id: str,
    topic: str,
    recommendation: str,
    options: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "id": decision_id,
        "topic": topic,
        "status": "pending",
        "recommendation": recommendation,
        "options": options,
        "reason": reason,
    }


def fixed(topic: str, action: str, reason: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "status": "recommended-default",
        "recommendation": action,
        "options": [action],
        "reason": reason,
    }


def missing_decision(role: str, column: str, item: dict[str, Any]) -> dict[str, Any]:
    rate = float(item.get("missing_rate", 0.0) or 0.0)
    inferred = str(item.get("inferred_type", "unknown"))
    if rate <= 0:
        return fixed("missing-values", "no-action", "该字段没有缺失值。")
    if role == "outcome":
        options = ["complete-case", "model-specific-method"]
        recommendation = "complete-case"
        reason = "因变量存在缺失；不应默认填补。选择完整案例会删除这些记录，须同时查看各结果类别损失。"
    elif inferred in NUMERIC_TYPES:
        options = ["complete-case", "median-imputation", "missing-indicator", "model-specific-method"]
        recommendation = "median-imputation"
        reason = "数值自变量存在缺失；完整案例会缩小样本，单次中位数填补会低估填补不确定性。"
    elif inferred in CATEGORICAL_TYPES:
        options = ["complete-case", "most-frequent", "explicit-missing-category", "model-specific-method"]
        recommendation = "explicit-missing-category"
        reason = "分类字段存在缺失；完整案例会缩小样本，众数填补会增加最常见类别，单列缺失类别则改变解释对象。"
    else:
        options = ["complete-case", "model-specific-method"]
        recommendation = "model-specific-method"
        reason = "该字段类型不明确且存在缺失，不能自动决定填补方法。"
    return pending(
        f"missing:{column}", "missing-values", recommendation, options, reason
    )


def encoding_decision(role: str, column: str, item: dict[str, Any]) -> dict[str, Any]:
    inferred = str(item.get("inferred_type", "unknown"))
    unique_count = int(item.get("unique_count", 0) or 0)
    if inferred in {"categorical", "boolean"}:
        return fixed(
            "categorical-encoding",
            "treat-as-categorical",
            "本阶段只记录为分类字段；参考类别、正类和具体模型编码在模型确认阶段决定。",
        )
    if inferred == "text" and 1 < unique_count <= 100:
        return pending(
            f"encoding:{column}",
            "low-cardinality-text",
            "confirm-categorical-meaning",
            ["treat-as-categorical", "keep-as-text-and-exclude", "manual-recode"],
            f"文本字段只有 {unique_count} 个唯一值，可能实际代表分类变量；需要确认类别含义后再编码。",
        )
    if inferred in RISKY_TYPES:
        return pending(
            f"inclusion:{column}",
            "variable-inclusion",
            "exclude",
            ["exclude", "include-with-justification"],
            f"字段推断类型为 {inferred}，默认不应直接进入模型。",
        )
    return fixed("categorical-encoding", "not-required", "当前推断类型不需要分类编码。")


def type_decision(column: str, item: dict[str, Any]) -> dict[str, Any]:
    if item.get("mixed_python_types"):
        return pending(
            f"type:{column}",
            "type-conversion",
            "inspect-and-recode",
            ["inspect-and-recode", "coerce-invalid-to-missing", "exclude"],
            "字段包含混合类型值，必须确认合法格式和转换失败的处理方式。",
        )
    return fixed("type-conversion", "keep-current-type", "未检测到混合类型。")


def outlier_decision(column: str, item: dict[str, Any]) -> dict[str, Any]:
    numeric = item.get("numeric_summary") or {}
    count = int(numeric.get("iqr_outlier_count", 0) or 0)
    if count > 0:
        return pending(
            f"outlier:{column}",
            "candidate-outliers",
            "keep-and-run-diagnostics",
            ["keep-and-run-diagnostics", "transform", "winsorize", "exclude-with-domain-rule"],
            f"IQR 规则标记了 {count} 个候选异常值；自动阈值不能单独作为删除依据。",
        )
    return fixed("candidate-outliers", "keep", "IQR 规则未标记候选异常值。")


def scaling_decision(goal: str, column: str, item: dict[str, Any]) -> dict[str, Any]:
    inferred = str(item.get("inferred_type", "unknown"))
    if inferred not in NUMERIC_TYPES:
        return fixed("scaling", "not-applicable", "非数值字段不进行数值标准化。")
    if goal == "prediction":
        return fixed(
            "scaling",
            "standardize-inside-training-pipeline",
            "预测任务中的标准化必须只在训练数据上拟合，避免数据泄漏。",
        )
    return fixed(
        "scaling",
        "keep-original-scale",
        "解释性分析默认保留原量纲；如需比较效应，可额外报告标准化系数。",
    )


def variable_action(
    goal: str, role: str, column: str, item: dict[str, Any]
) -> dict[str, Any]:
    inclusion = "exclude-recommended" if item.get("constant") else "include-provisionally"
    warnings: list[str] = []
    if item.get("constant"):
        warnings.append("字段为常量或全空，不能提供模型信息。")
    if item.get("near_constant"):
        warnings.append("字段接近常量，可能造成估计不稳定。")
    if item.get("sensitive_name_candidate"):
        warnings.append("字段疑似敏感，需要确认分析必要性和隐私处理。")
    return {
        "role": role,
        "column": column,
        "inferred_type": item.get("inferred_type"),
        "missing_rate": float(item.get("missing_rate", 0.0) or 0.0),
        "inclusion": inclusion,
        "missing": missing_decision(role, column, item),
        "type_conversion": type_decision(column, item),
        "encoding": encoding_decision(role, column, item),
        "outliers": outlier_decision(column, item),
        "scaling": scaling_decision(goal, column, item),
        "warnings": warnings,
    }


def collect_pending(row_actions: list[dict[str, Any]], variable_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for action in row_actions:
        if action.get("status") == "pending":
            decisions.append(action)
    for variable in variable_actions:
        for key in ("missing", "type_conversion", "encoding", "outliers", "scaling"):
            decision = variable[key]
            if decision.get("status") == "pending":
                decisions.append({"column": variable["column"], "role": variable["role"], **decision})
    return decisions


SUPPORTED_EXECUTION_OPTIONS = {
    "rows:": {"keep", "drop-exact-duplicates", "deduplicate-with-key", "drop-fully-empty-rows"},
    "missing:": {
        "complete-case",
        "median-imputation",
        "missing-indicator",
        "most-frequent",
        "explicit-missing-category",
        "model-specific-method",
    },
    "type:": {"inspect-and-recode", "coerce-invalid-to-missing", "exclude"},
    "encoding:": {"treat-as-categorical", "keep-as-text-and-exclude", "manual-recode"},
    "inclusion:": {"exclude", "include-with-justification"},
    "outlier:": {"keep-and-run-diagnostics", "transform", "winsorize", "exclude-with-domain-rule"},
}


def validate_execution_contract(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    for decision in decisions:
        decision_id = str(decision.get("id", ""))
        prefix = next((value for value in SUPPORTED_EXECUTION_OPTIONS if decision_id.startswith(value)), None)
        if prefix is None:
            raise ValueError(f"Preparation decision {decision_id!r} has no execution handler")
        options = decision.get("options")
        if not isinstance(options, list):
            raise ValueError(f"Preparation decision {decision_id!r} has no options array")
        unsupported = [option for option in options if option not in SUPPORTED_EXECUTION_OPTIONS[prefix]]
        if unsupported:
            raise ValueError(
                f"Preparation decision {decision_id!r} offers unsupported choices: {unsupported}"
            )
    return {"status": "compatible", "validated_pending_decisions": len(decisions)}


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# 数据处理方案",
        "",
        f"- 状态：`{plan['status']}`",
        f"- 源文件：`{plan['source_file']}`",
        f"- 工作表：`{plan['sheet']}`",
        f"- 分析目标：`{plan['goal']}`",
        f"- 原始数据是否修改：{'否' if plan['raw_data_unchanged'] else '是'}",
        f"- 待确认决策：{len(plan['pending_decisions'])} 项",
        "",
        "## 缺失值处理前的样本构成检查",
        "",
        f"- 若对所有入模字段做完整案例处理，将保留 {plan['missingness_bias_screen']['complete_case_rows']} 行，排除 {plan['missingness_bias_screen']['complete_case_excluded_rows']} 行（{plan['missingness_bias_screen']['complete_case_excluded_rate']:.1%}）。",
        "- 该检查只描述缺失组与非缺失组的结果分布差异，不能证明缺失机制，也不是模型系数的敏感性分析。",
        "- 如需判断模型结论是否随缺失处理方法变化，必须进入专门的模型级敏感性流程；当前单一 cleaned-data.csv 流程不能替代该分析。",
        "",
        "## 行级处理",
        "",
    ]
    for action in plan["row_actions"]:
        lines.extend(
            [
                f"### {action['topic']}",
                f"- 状态：`{action['status']}`",
                f"- 推荐：`{action['recommendation']}`",
                f"- 理由：{action['reason']}",
                f"- 可选方案：{', '.join(action['options'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## 变量处理概览",
            "",
            "| 角色 | 字段 | 类型 | 缺失率 | 暂定纳入 | 缺失处理 | 编码 | 异常值 | 标准化 |",
            "|---|---|---|---:|---|---|---|---|---|",
        ]
    )
    for action in plan["variable_actions"]:
        lines.append(
            f"| {action['role']} | `{action['column']}` | {action['inferred_type']} | "
            f"{action['missing_rate']:.1%} | {action['inclusion']} | "
            f"{action['missing']['recommendation']} | {action['encoding']['recommendation']} | "
            f"{action['outliers']['recommendation']} | {action['scaling']['recommendation']} |"
        )
    lines.extend(["", "## 待确认决策", ""])
    if plan["pending_decisions"]:
        for index, decision in enumerate(plan["pending_decisions"], 1):
            subject = f"`{decision['column']}`：" if decision.get("column") else ""
            lines.extend(
                [
                    f"{index}. {subject}{decision['topic']}",
                    f"   - 推荐：`{decision['recommendation']}`",
                    f"   - 理由：{decision['reason']}",
                    f"   - 选项：{', '.join(decision['options'])}",
                ]
            )
    else:
        lines.append("- 没有自动识别出的待决策项；展示执行方案后可直接进入处理，无需再次整体确认。")
    lines.extend(
        [
            "",
            "## 确认状态",
            "",
            "- 当前计划仍是草稿。",
            "- 用户确认前不得修改数据或生成 cleaned-data.csv。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        profile = read_json(args.profile.resolve(), "profile")
        task = read_json(args.task.resolve(), "analysis task")
        input_path = args.input.resolve()
        if not input_path.is_file():
            raise ValueError(f"Source table not found: {input_path}")
        if task.get("status") != "approved" or not isinstance(task.get("approval"), dict):
            raise ValueError(
                "Source analysis task is not backed by a questionnaire approval receipt. "
                "Run approve_analysis_task.py first."
            )
        if task.get("requires_user_confirmation") is not False:
            raise ValueError("Approved analysis task still requires user confirmation")
        sheet = choose_profile_sheet(profile, str(task.get("sheet", "")))
        table_read_spec = confirmed_table_read_spec(profile, str(task.get("sheet", "")))
        if task.get("table_read_spec") != table_read_spec:
            raise ValueError(
                "Analysis task and profile use different table-read specifications; regenerate the task card"
            )
        columns = {str(item["column"]): item for item in sheet.get("columns", [])}

        role_pairs = [("outcome", str(task.get("outcome", "")))]
        role_pairs.extend(("predictor", str(name)) for name in task.get("predictors", []))
        role_pairs.extend(("control", str(name)) for name in task.get("controls", []))
        missing_columns = [name for _, name in role_pairs if name not in columns]
        if missing_columns:
            raise ValueError(
                "Task variables are absent from the profile: " + ", ".join(missing_columns)
            )

        row_actions: list[dict[str, Any]] = []
        duplicate_count = int(sheet.get("duplicate_row_count", 0) or 0)
        if duplicate_count:
            row_actions.append(
                pending(
                    "rows:duplicates",
                    "exact-duplicate-rows",
                    "inspect-then-decide",
                    ["keep", "drop-exact-duplicates", "deduplicate-with-key"],
                    f"检测到 {duplicate_count} 条完全重复记录；需确认它们是错误重复还是合法重复事件。",
                )
            )
        else:
            row_actions.append(fixed("exact-duplicate-rows", "keep", "未检测到完全重复记录。"))
        empty_rows = int(sheet.get("fully_empty_row_count", 0) or 0)
        if empty_rows:
            row_actions.append(
                pending(
                    "rows:empty",
                    "fully-empty-rows",
                    "drop-fully-empty-rows",
                    ["drop-fully-empty-rows", "keep"],
                    f"检测到 {empty_rows} 条全空记录；通常可删除，但仍需记录并确认。",
                )
            )
        else:
            row_actions.append(fixed("fully-empty-rows", "keep", "未检测到全空记录。"))

        variable_actions = [
            variable_action(str(task.get("goal")), role, column, columns[column])
            for role, column in role_pairs
        ]
        variables = [
            {
                "role": action["role"],
                "column": action["column"],
                "inferred_type": str(action.get("inferred_type", "unknown")),
            }
            for action in variable_actions
        ]
        source_hash = file_sha256(input_path)
        if table_read_spec.get("source_sha256") != source_hash:
            raise ValueError("Source file hash differs from the confirmed table-read specification")
        frame, _loaded_sheet = load_table(
            input_path, {"table_read_spec": table_read_spec}, None
        )
        missingness_screen = build_missingness_screen(frame, variables)
        pending_decisions = collect_pending(row_actions, variable_actions)
        execution_contract = validate_execution_contract(pending_decisions)
        warnings = [
            f"{item['role']} `{item['column']}`：{warning}"
            for item in variable_actions
            for warning in item["warnings"]
        ]
        plan = {
            "status": "draft",
            "requires_user_confirmation": True,
            "source_task_confirmation": {
                "recorded": True,
                "source_task_status": task.get("status"),
                "recorded_at": datetime.now().astimezone().isoformat(),
            },
            "source_file": profile.get("source_file"),
            "sheet": task.get("sheet"),
            "table_read_spec": table_read_spec,
            "goal": task.get("goal"),
            "unit_of_analysis": task.get("unit_of_analysis"),
            "raw_data_unchanged": True,
            "missingness_bias_screen": missingness_screen,
            "missingness_bias_rows": flatten_missingness_screen(missingness_screen),
            "missingness_conclusion_contract": {
                "status": "restricted" if missingness_screen["status"] == "review-required" else "clear",
                "scope": "analyzed-sample-only" if missingness_screen["status"] == "review-required" else "full-selected-sample",
                "model_estimate_sensitivity_completed": False,
                "required_statement": "存在所选字段缺失时，结论只适用于实际进入模型的样本；不得声称缺失是随机的。",
            },
            "row_actions": row_actions,
            "variable_actions": variable_actions,
            "warnings": warnings,
            "pending_decisions": pending_decisions,
            "execution_contract": execution_contract,
            "created_at": datetime.now().astimezone().isoformat(),
        }

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "data-preparation-plan.json"
        md_path = output_dir / "data-preparation-plan.md"
        json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(plan), encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "draft",
                    "pending_decisions": len(pending_decisions),
                    "outputs": [str(md_path), str(json_path)],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except ValueError as exc:
        print(f"Preparation planning failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
