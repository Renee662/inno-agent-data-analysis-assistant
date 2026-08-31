#!/usr/bin/env python3
"""Build a validated draft analysis task card from a tabular data profile."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


GOALS = ("association", "prediction", "description", "causal")
DECISION_GOALS = ("relationships", "prediction", "group-comparison")
RISKY_TYPES = {"identifier-candidate", "sensitive-candidate", "all-missing"}
SEMANTIC_BATCH_SIZE = 3
MAX_STRUCTURAL_CANDIDATES = 3
STRUCTURAL_NAME_RE = re.compile(
    r"(^|[_\-\s])(iid|pid|id|subject|participant|person|partner|student|patient|"
    r"user|household|school|class|group|cluster|site|wave|batch|round|time|date|year)"
    r"([_\-\s]|$)|参与者|对象|个体|配对|学校|班级|组别|批次|轮次|时间|日期",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Path to data-profile.json")
    parser.add_argument(
        "--context",
        type=Path,
        help="Optional dataset-context.json produced by tabular-data-profiler",
    )
    parser.add_argument(
        "--public-verification",
        type=Path,
        help="Optional public-dataset-verification.json from source verification",
    )
    parser.add_argument("--goal", choices=GOALS, required=True)
    parser.add_argument("--decision-goal", choices=DECISION_GOALS)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--predictors", nargs="+", required=True)
    parser.add_argument("--controls", nargs="*", default=[])
    parser.add_argument("--sheet")
    parser.add_argument("--unit-of-analysis")
    parser.add_argument("--title", default="变量关系分析")
    parser.add_argument("--output-dir", type=Path, default=Path("work/analysis-plan"))
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"Profile file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read profile JSON: {exc}") from exc
    if not isinstance(payload.get("profiles"), list) or not payload["profiles"]:
        raise ValueError("Profile JSON contains no profiled sheets")
    return payload


def load_context(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists() or not path.is_file():
        raise ValueError(f"Dataset context file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read dataset context JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Dataset context JSON must contain an object")
    return payload


def choose_sheet(payload: dict[str, Any], requested: str | None) -> dict[str, Any]:
    profiles = payload["profiles"]
    names = [str(item.get("sheet", "")) for item in profiles]
    if requested:
        for profile in profiles:
            if profile.get("sheet") == requested:
                return profile
        raise ValueError(f"Sheet {requested!r} not found. Available sheets: {names}")
    if len(profiles) == 1:
        return profiles[0]
    raise ValueError(f"Multiple sheets were profiled; choose one with --sheet. Available sheets: {names}")


def confirmed_table_read_spec(payload: dict[str, Any], sheet_name: str) -> dict[str, Any]:
    document = payload.get("table_read_spec")
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(
            "Profile has no supported table-read specification. Re-run tabular profiling before planning."
        )
    sheets = document.get("sheets")
    if not isinstance(sheets, list):
        raise ValueError("Profile table-read specification has no sheets array")
    selected_sheet = next(
        (item for item in sheets if isinstance(item, dict) and item.get("sheet") == sheet_name),
        None,
    )
    if not isinstance(selected_sheet, dict):
        raise ValueError(f"Table-read specification is missing sheet {sheet_name!r}")
    if selected_sheet.get("requires_user_confirmation") is True or selected_sheet.get("status") not in {
        "auto-confirmed",
        "user-confirmed",
    }:
        selected = selected_sheet.get("selected", {})
        rows = selected.get("header_rows", []) if isinstance(selected, dict) else []
        raise ValueError(
            f"Table structure confirmation required for sheet {sheet_name!r}; "
            f"current candidate header rows are {rows}. Confirm the preview and re-run profiling."
        )
    selected = selected_sheet.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"Confirmed table-read specification for {sheet_name!r} is invalid")
    return selected


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def validate_roles(
    sheet: dict[str, Any], outcome: str, predictors: list[str], controls: list[str]
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    columns = {str(item["column"]): item for item in sheet.get("columns", [])}
    requested = [outcome, *predictors, *controls]
    missing = [name for name in requested if name not in columns]
    if missing:
        suggestions = []
        for name in missing:
            matches = difflib.get_close_matches(name, list(columns), n=3, cutoff=0.45)
            suggestions.append(f"{name!r}: {matches or 'no close match'}")
        raise ValueError(
            "Unknown columns: " + ", ".join(repr(name) for name in missing)
            + ". Suggestions: " + "; ".join(suggestions)
            + ". Available columns: " + ", ".join(columns)
        )
    conflicts = sorted(set(predictors) & set(controls))
    if outcome in predictors or outcome in controls:
        conflicts.append(outcome)
    if conflicts:
        raise ValueError("Variables cannot have multiple roles: " + ", ".join(sorted(set(conflicts))))
    return columns, predictors, controls


def variable_card(role: str, name: str, profile: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    inferred = str(profile.get("inferred_type", "unknown"))
    missing_rate = float(profile.get("missing_rate", 0.0) or 0.0)
    if inferred in RISKY_TYPES:
        warnings.append(f"字段类型为 {inferred}，需要确认是否适合参与分析")
    if profile.get("sensitive_name_candidate"):
        warnings.append("疑似敏感字段，需要说明必要性并避免泄露")
    if profile.get("constant"):
        warnings.append("字段为常量或全空，不能提供模型信息")
    elif profile.get("near_constant"):
        warnings.append("字段接近常量，可能导致估计不稳定")
    if profile.get("mixed_python_types"):
        warnings.append("字段包含混合类型值，需要先清理或确认编码")
    if missing_rate > 0:
        warnings.append(f"缺失率为 {missing_rate:.1%}，需要确认处理方法")
    return {
        "role": role,
        "column": name,
        "inferred_type": inferred,
        "pandas_dtype": profile.get("pandas_dtype"),
        "missing_rate": missing_rate,
        "unique_count": int(profile.get("unique_count", 0) or 0),
        "warnings": warnings,
    }


def excerpt_near_column(text: str, column: str) -> str | None:
    if not text:
        return None
    pattern = re.compile(rf"(?<![\w]){re.escape(column)}(?![\w])", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    start = max(0, match.start() - 140)
    end = min(len(text), match.end() + 220)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet or None


def context_evidence(
    column: str,
    context: dict[str, Any],
    public_verification: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in context.get("companion_files", []):
        if not isinstance(item, dict) or column not in item.get("column_mentions", []):
            continue
        evidence.append(
            {
                "source_type": "companion-file",
                "source_path": item.get("path"),
                "category": item.get("category"),
                "extraction_status": item.get("extraction_status"),
                "title_hints": item.get("title_hints", [])[:3],
                "snippet": excerpt_near_column(str(item.get("excerpt", "")), column),
            }
        )
        if len(evidence) >= 5:
            return evidence

    for sheet in context.get("embedded_context_sheets", []):
        if not isinstance(sheet, dict):
            continue
        for row in sheet.get("preview", []):
            if not isinstance(row, dict):
                continue
            if not any(str(value).strip().casefold() == column.casefold() for value in row.values()):
                continue
            evidence.append(
                {
                    "source_type": "workbook-context-sheet",
                    "source_path": f"[sheet:{sheet.get('sheet')}]",
                    "category": "codebook-candidate",
                    "declared_values": row,
                }
            )
            break
        if len(evidence) >= 5:
            break

    for candidate in public_verification.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        comparison = candidate.get("comparison", {})
        if not isinstance(comparison, dict):
            continue
        if comparison.get("candidate_status") == "insufficient-column-match":
            continue
        reference_columns = {
            str(value).casefold()
            for value in candidate.get("columns", [])
            if isinstance(value, str)
        }
        if column.casefold() not in reference_columns:
            continue
        field_definitions = candidate.get("field_definitions", {})
        definitions_by_name = (
            {str(key).casefold(): value for key, value in field_definitions.items()}
            if isinstance(field_definitions, dict)
            else {}
        )
        evidence.append(
            {
                "source_type": "public-dataset-candidate",
                "source_path": candidate.get("source_url"),
                "source_title": candidate.get("source_title"),
                "publisher": candidate.get("publisher"),
                "dataset_id": candidate.get("dataset_id"),
                "version": candidate.get("version"),
                "candidate_status": comparison.get("candidate_status"),
                "identity_confirmed": False,
                "declared_values": definitions_by_name.get(column.casefold(), {}),
                "snippet": excerpt_near_column(str(candidate.get("description", "")), column),
            }
        )
        if len(evidence) >= 5:
            break
    return evidence


def structural_candidates(
    sheet: dict[str, Any], selected_columns: set[str]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for column in sheet.get("columns", []):
        if not isinstance(column, dict):
            continue
        name = str(column.get("column", ""))
        if not name or name in selected_columns:
            continue
        inferred = str(column.get("inferred_type", ""))
        non_null = int(column.get("non_null_count", 0) or 0)
        unique = int(column.get("unique_count", 0) or 0)
        repeated = 0 < unique < non_null
        name_match = bool(STRUCTURAL_NAME_RE.search(name))
        identifier = inferred == "identifier-candidate"
        datetime_like = inferred == "datetime"
        if not name_match and not identifier and not datetime_like:
            continue
        score = (4 if name_match else 0) + (3 if identifier else 0) + (2 if repeated else 0)
        candidates.append(
            {
                "column": name,
                "inferred_type": inferred,
                "unique_count": unique,
                "non_null_count": non_null,
                "repeated_values": repeated,
                "reason": (
                    "字段名称或类型表明它可能标识个体、配对、组别、批次或时间；"
                    "若同一值重复出现，可能改变每行独立性的判断。"
                ),
                "_score": score,
            }
        )
    candidates.sort(key=lambda item: (-item["_score"], item["column"].casefold()))
    for item in candidates:
        item.pop("_score", None)
    return candidates[:MAX_STRUCTURAL_CANDIDATES]


def build_semantic_review(
    variables: list[dict[str, Any]],
    sheet: dict[str, Any],
    context: dict[str, Any],
    public_verification: dict[str, Any],
    unit_of_analysis: str | None,
) -> dict[str, Any]:
    fields = []
    for priority, variable in enumerate(variables):
        column = variable["column"]
        evidence = context_evidence(column, context, public_verification)
        fields.append(
            {
                "column": column,
                "role": variable["role"],
                "priority": priority,
                "semantic_status": "candidate-from-local-source" if evidence else "unresolved",
                "source_evidence": evidence,
                "requires_confirmation": True,
                "interaction_template": {
                    "candidate_label": (
                        "采用本地证据中的候选含义"
                        if evidence
                        else "采用 Agent 候选推测（需明确核对）"
                    ),
                    "correction_label": "这个含义不对，我来补充",
                    "help_label": f"我不确定 {column} 的候选含义为什么这样判断",
                    "custom_input_required_for_correction": True,
                    "help_or_objection_is_confirmation": False,
                },
                "required_metadata": [
                    "display_name",
                    "meaning",
                    "unit_or_category_meaning",
                    "semantic_status",
                    "source_type",
                    "source_path",
                    "user_confirmed",
                ],
            }
        )
    batches = [
        [item["column"] for item in fields[index : index + SEMANTIC_BATCH_SIZE]]
        for index in range(0, len(fields), SEMANTIC_BATCH_SIZE)
    ]
    selected = {item["column"] for item in variables}
    return {
        "status": "pending",
        "selected_fields_only": True,
        "question_batch_size": SEMANTIC_BATCH_SIZE,
        "fields": fields,
        "question_batches": batches,
        "unit_of_analysis_confirmed": bool(unit_of_analysis),
        "structural_candidates": structural_candidates(sheet, selected),
        "approval_blocked_until_resolved": True,
        "rules": [
            "Do not ask about unselected ordinary columns.",
            "Ask at most three selected fields per questionnaire.",
            "Use source-declared meanings as candidates, not automatic confirmation.",
            "A help, uncertainty, or custom objection answer does not confirm semantics.",
        ],
    }


def recommend_method(goal: str, outcome: dict[str, Any]) -> dict[str, str]:
    inferred = outcome["inferred_type"]
    unique_count = outcome["unique_count"]
    if goal == "causal":
        return {
            "method": "causal-design-review",
            "reason": "因果问题需要先确认时间顺序、处理变量、混杂因素和识别策略；普通回归不足以证明因果。",
        }
    if goal == "description":
        return {
            "method": "descriptive-analysis",
            "reason": "当前目标是描述数据，不应在用户确认前自动升级为推断或预测模型。",
        }
    if inferred in {"boolean", "categorical"} and unique_count == 2:
        method = "binary-classification" if goal == "prediction" else "binary-logistic-regression"
        return {"method": method, "reason": "因变量为二分类变量。"}
    if inferred == "categorical" and unique_count > 2:
        method = "multiclass-classification" if goal == "prediction" else "multinomial-or-ordinal-logistic-review"
        return {
            "method": method,
            "reason": "因变量有三个及以上类别；需要确认类别是否存在自然顺序，再选择多分类或有序 Logistic 回归。",
        }
    if inferred == "numeric-continuous":
        method = "regression-with-validation" if goal == "prediction" else "multiple-linear-regression"
        return {"method": method, "reason": "因变量为连续数值变量。"}
    if inferred == "numeric-discrete":
        return {
            "method": "linear-or-count-model-review",
            "reason": "因变量为离散数值；需要确认它是计数、等级还是近似连续变量后才能确定模型。",
        }
    return {
        "method": "manual-outcome-type-review",
        "reason": f"因变量推断类型为 {inferred}，无法自动确定合适模型。",
    }


def build_questions(
    goal: str,
    unit: str | None,
    variables: list[dict[str, Any]],
    controls: list[str],
    recommendation: dict[str, str],
) -> list[str]:
    questions = [
        "请确认因变量、自变量和控制变量的业务含义是否正确。",
        "每一行是否代表相互独立的观测？是否存在同一对象重复测量、班级/机构嵌套或时间序列？",
    ]
    if not unit:
        questions.insert(0, "每一行代表什么分析单位？例如一名学生、一次测量或一个机构。")
    if not controls:
        questions.append("是否有基于业务或理论应当控制的混杂变量？不要仅凭显著性选择控制变量。")
    if any(variable["missing_rate"] > 0 for variable in variables):
        questions.append("所选变量存在缺失值：后续应删除、填补还是使用其他策略？")
    if any(variable["warnings"] for variable in variables):
        questions.append("请逐项确认任务卡中的字段风险和数据质量警告。")
    if goal == "causal":
        questions.append("因果目标中的处理/暴露变量是什么？时间顺序、混杂因素和识别假设是什么？")
    if recommendation["method"] in {"linear-or-count-model-review", "manual-outcome-type-review"}:
        questions.append("请确认因变量的测量尺度和允许取值，以便最终选择模型。")
    if recommendation["method"] == "multinomial-or-ordinal-logistic-review":
        questions.append(
            "这些结果类别是否存在明确的自然顺序？若有，请从最低到最高列出全部类别；若没有，请确认参照类别。"
        )
    return questions


def render_markdown(task: dict[str, Any]) -> str:
    lines = [
        f"# {task['title']}",
        "",
        f"- 状态：`{task['status']}`",
        f"- 数据文件：`{task['source_file']}`",
        f"- 工作表：`{task['sheet']}`",
            f"- 分析目标：`{task['goal']}`",
            f"- 用户决策目标：`{task.get('decision_goal') or '待确认'}`",
        f"- 分析单位：{task['unit_of_analysis'] or '待确认'}",
        "",
        "## 变量角色",
        "",
        "| 角色 | 字段 | 推断类型 | 缺失率 | 唯一值 |",
        "|---|---|---|---:|---:|",
    ]
    for variable in task["variables"]:
        lines.append(
            f"| {variable['role']} | `{variable['column']}` | {variable['inferred_type']} | "
            f"{variable['missing_rate']:.1%} | {variable['unique_count']} |"
        )
    lines.extend(
        [
            "",
            "## 候选方法",
            "",
            f"- 推荐：`{task['recommendation']['method']}`",
            f"- 理由：{task['recommendation']['reason']}",
            "",
            "该方法只是候选方案，得到用户确认后才能成为最终方案。",
            "",
            "## 警告",
            "",
        ]
    )
    if task["warnings"]:
        lines.extend(f"- {warning}" for warning in task["warnings"])
    else:
        lines.append("- 当前所选字段未触发自动风险规则。")
    lines.extend(
        [
            "",
            "## 字段语义确认队列",
            "",
            "| 优先级 | 角色 | 字段 | 当前状态 | 本地证据数 |",
            "|---:|---|---|---|---:|",
        ]
    )
    for field in task["semantic_review"]["fields"]:
        lines.append(
            f"| {field['priority'] + 1} | {field['role']} | `{field['column']}` | "
            f"{field['semantic_status']} | {len(field['source_evidence'])} |"
        )
    lines.extend(["", "每次结构化问卷最多询问三个上述字段。"])
    if task["semantic_review"]["structural_candidates"]:
        lines.extend(["", "### 可能影响分析单位的字段", ""])
        for candidate in task["semantic_review"]["structural_candidates"]:
            repeated = "存在重复值" if candidate["repeated_values"] else "未发现重复值"
            lines.append(f"- `{candidate['column']}`：{candidate['reason']}（{repeated}）")
    lines.extend(["", "## 待确认", ""])
    lines.extend(f"{index}. {question}" for index, question in enumerate(task["questions"], 1))
    lines.extend(
        [
            "",
            "## 确认状态",
            "",
            "- 当前状态：草稿",
            "- 在用户明确批准或修改前，不得清洗数据或运行模型。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        payload = load_profile(args.profile.resolve())
        context = load_context(args.context.resolve() if args.context else None)
        public_verification = load_context(
            args.public_verification.resolve() if args.public_verification else None
        )
        sheet = choose_sheet(payload, args.sheet)
        table_read_spec = confirmed_table_read_spec(payload, str(sheet.get("sheet", "")))
        predictors = dedupe(args.predictors)
        controls = dedupe(args.controls)
        columns, predictors, controls = validate_roles(sheet, args.outcome, predictors, controls)

        variables = [variable_card("outcome", args.outcome, columns[args.outcome])]
        variables.extend(variable_card("predictor", name, columns[name]) for name in predictors)
        variables.extend(variable_card("control", name, columns[name]) for name in controls)
        recommendation = recommend_method(args.goal, variables[0])
        semantic_review = build_semantic_review(
            variables, sheet, context, public_verification, args.unit_of_analysis
        )
        warnings = [
            f"{variable['role']} `{variable['column']}`：{warning}"
            for variable in variables
            for warning in variable["warnings"]
        ]
        if args.goal == "causal":
            warnings.append("当前任务具有因果意图；在识别策略确认前不能把回归关联解释为因果效应。")
        questions = build_questions(
            args.goal, args.unit_of_analysis, variables, controls, recommendation
        )
        task = {
            "title": args.title,
            "status": "draft",
            "requires_user_confirmation": True,
            "source_file": payload.get("source_file"),
            "profile_file": str(args.profile),
            "sheet": sheet.get("sheet"),
            "table_read_spec": table_read_spec,
            "goal": args.goal,
            "decision_goal": args.decision_goal,
            "unit_of_analysis": args.unit_of_analysis,
            "outcome": args.outcome,
            "predictors": predictors,
            "controls": controls,
            "variables": variables,
            "recommendation": recommendation,
            "dataset_context_file": str(args.context) if args.context else None,
            "public_dataset_verification_file": (
                str(args.public_verification) if args.public_verification else None
            ),
            "semantic_review": semantic_review,
            "warnings": warnings,
            "questions": questions,
            "created_at": datetime.now().astimezone().isoformat(),
        }

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "analysis-task.json"
        md_path = output_dir / "analysis-plan.md"
        json_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_markdown(task), encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "draft",
                    "method": recommendation["method"],
                    "outputs": [str(md_path), str(json_path)],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except ValueError as exc:
        print(f"Planning validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
