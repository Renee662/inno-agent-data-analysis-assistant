#!/usr/bin/env python3
"""Create a binding pre-model assessment of whether the bundled workflow fits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from approval import ApprovalError, approval_summary, verify_approval  # noqa: E402
from file_utils import atomic_replace_text, sha256_file  # noqa: E402
from model_registry import SUPPORTED_MODEL_TYPES  # noqa: E402


SUPPORTED_MODELS = list(SUPPORTED_MODEL_TYPES)


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--decision-input", required=True)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Analysis task does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Cannot read analysis task: {exc}")
    if not isinstance(value, dict):
        fail("Analysis task must contain a JSON object")
    return value


def atomic_write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"Output already exists: {path}")
    atomic_replace_text(path, content)


def blocker(
    code: str,
    evidence: str,
    required_workflow: str,
    mismatch: str,
) -> dict[str, str]:
    return {
        "code": code,
        "evidence": evidence,
        "required_workflow": required_workflow,
        "why_existing_models_mismatch": mismatch,
    }


def main() -> None:
    args = parse_args()

    task_path = Path(args.task).resolve()
    task = read_json(task_path)
    if task.get("status") != "approved" or not isinstance(task.get("approval"), dict):
        fail("Workflow assessment requires approved-analysis-task.json")
    decision_path = Path(args.decision_input).resolve()
    decision = read_json(decision_path)
    try:
        approval_record = verify_approval(
            Path(args.approval_record).resolve(),
            "approve-workflow-support",
            decision_path,
        )
    except ApprovalError as exc:
        fail(str(exc))
    allowed = {
        "observation_structure": {"independent", "repeated", "nested", "paired", "time-series", "unknown"},
        "outcome_process": {"standard", "time-to-event", "censored", "zero-inflated", "unknown"},
        "sampling_design": {"simple", "weighted", "clustered", "stratified", "complex", "unknown"},
        "count_exposure": {"not-applicable", "none", "required", "unknown"},
    }
    for field, choices in allowed.items():
        value = decision.get(field)
        if value not in choices:
            fail(f"Workflow decision input has invalid {field}: {value!r}")
        setattr(args, field, value)
    goal = str(task.get("goal", "")).strip().casefold()
    decision_goal = str(task.get("decision_goal", "")).strip().casefold()
    if not goal:
        fail("Analysis task must contain a goal")

    checked = {
        "goal": goal,
        "decision_goal": decision_goal or None,
        "observation_structure": args.observation_structure,
        "outcome_process": args.outcome_process,
        "sampling_design": args.sampling_design,
        "count_exposure": args.count_exposure,
    }
    unknown_dimensions = [
        label
        for label, value in (
            ("记录之间的关系", args.observation_structure),
            ("结果产生过程", args.outcome_process),
            ("抽样设计", args.sampling_design),
            ("计数暴露量/观察时长", args.count_exposure),
        )
        if value == "unknown"
    ]
    blockers: list[dict[str, str]] = []

    if goal == "causal":
        blockers.append(
            blocker(
                "causal-identification-required",
                "分析任务的目标是因果推断",
                "因果识别与敏感性分析工作流",
                "现有回归只能估计条件关联，没有处理分配机制、时间顺序、未测混杂和识别假设。",
            )
        )
    # Prediction is supported by the bundled fixed-specification cross-validation
    # path. It estimates internal sample-out performance without claiming external
    # validation or performing outcome-guided model/feature selection.
    if goal == "description":
        blockers.append(
            blocker(
                "descriptive-goal-does-not-require-regression",
                "分析任务仅要求描述数据",
                "描述统计与可视化工作流",
                "强行拟合回归会回答用户没有提出的问题。",
            )
        )

    observation_rules = {
        "repeated": (
            "同一对象存在重复测量",
            "混合效应模型或 GEE 工作流",
            "HC3 标准误不能处理同一对象内部相关性。",
        ),
        "nested": (
            "记录存在班级、机构、地区等嵌套层级",
            "多层/混合效应模型工作流",
            "普通单层回归会忽略组内相关和层级方差。",
        ),
        "paired": (
            "记录是配对或匹配观测",
            "配对分析、条件模型或匹配设计工作流",
            "把配对记录当作独立行会低估或错误估计不确定性。",
        ),
        "time-series": (
            "记录按时间连续排列并可能自相关",
            "时间序列或纵向分析工作流",
            "普通回归没有建模自相关、趋势、季节性和预测切分。",
        ),
    }
    if args.observation_structure in observation_rules:
        evidence, workflow, mismatch = observation_rules[args.observation_structure]
        blockers.append(
            blocker(
                f"unsupported-observation-structure:{args.observation_structure}",
                evidence,
                workflow,
                mismatch,
            )
        )

    outcome_rules = {
        "time-to-event": (
            "结果是事件发生时间",
            "生存分析工作流",
            "现有模型没有风险集、随访时间或删失机制。",
        ),
        "censored": (
            "结果含删失或截尾",
            "生存/删失回归工作流",
            "把删失值当作普通观测会造成有系统的偏差。",
        ),
        "zero-inflated": (
            "计数结果包含结构性零值机制",
            "零膨胀或 hurdle 计数模型工作流",
            "Poisson/负二项单过程模型不能区分结构性零与计数过程。",
        ),
    }
    if args.outcome_process in outcome_rules:
        evidence, workflow, mismatch = outcome_rules[args.outcome_process]
        blockers.append(
            blocker(
                f"unsupported-outcome-process:{args.outcome_process}",
                evidence,
                workflow,
                mismatch,
            )
        )

    if args.sampling_design != "simple" and args.sampling_design != "unknown":
        blockers.append(
            blocker(
                f"unsupported-sampling-design:{args.sampling_design}",
                f"抽样设计被确认为 {args.sampling_design}",
                "复杂抽样/调查加权工作流",
                "现有模型没有抽样权重、分层、聚类或有限总体修正。",
            )
        )
    if args.count_exposure == "required":
        blockers.append(
            blocker(
                "count-offset-required",
                "计数结果需要按暴露量、人口数或观察时长建模",
                "带 offset/exposure 的计数模型工作流",
                "现有 Poisson 和负二项执行器没有 offset/exposure 参数，直接拟合会把率误当作原始计数。",
            )
        )

    if blockers:
        status = "specialized-workflow-required"
        execution_allowed = False
        next_step = "停止现有模型流程；向用户说明不匹配证据和所需专门工作流。"
    elif unknown_dimensions:
        status = "needs-user-information"
        execution_allowed = False
        next_step = "只追问仍未知且会影响方法选择的结构信息，然后重新评估。"
    else:
        status = "supported"
        execution_allowed = True
        next_step = "可以进入具体模型规格确认；仍需通过模型类型、样本量和可识别性检查。"

    assessed_at = datetime.now().astimezone().isoformat()
    assessment = {
        "status": status,
        "execution_allowed": execution_allowed,
        "user_confirmed": True,
        "approval": approval_summary(approval_record, Path(args.approval_record).resolve()),
        "assessed_at": assessed_at,
        "supported_models": SUPPORTED_MODELS,
        "checked_dimensions": checked,
        "unknown_dimensions": unknown_dimensions,
        "blocking_reasons": blockers,
        "recommended_next_step": next_step,
        "analysis_task": str(task_path),
        "analysis_task_sha256": sha256_file(task_path),
    }
    lines = [
        "# 工作流适配性评估",
        "",
        f"- 状态：`{status}`",
        f"- 允许执行现有模型：{'是' if execution_allowed else '否'}",
        f"- 记录关系：`{args.observation_structure}`",
        f"- 结果过程：`{args.outcome_process}`",
        f"- 抽样设计：`{args.sampling_design}`",
        f"- 计数暴露量：`{args.count_exposure}`",
        "",
    ]
    if unknown_dimensions:
        lines.extend(["## 仍需了解", "", *(f"- {item}" for item in unknown_dimensions), ""])
    if blockers:
        lines.extend(["## 停止原因", ""])
        for item in blockers:
            lines.extend(
                [
                    f"### {item['required_workflow']}",
                    "",
                    f"- 证据：{item['evidence']}",
                    f"- 为什么现有模型不匹配：{item['why_existing_models_mismatch']}",
                    "",
                ]
            )
    lines.extend(["## 下一步", "", f"- {next_step}", ""])

    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "workflow-support-assessment.json"
    md_path = output_dir / "workflow-support-assessment.md"
    atomic_write(
        json_path,
        json.dumps(assessment, ensure_ascii=False, indent=2) + "\n",
        args.overwrite,
    )
    atomic_write(md_path, "\n".join(lines), args.overwrite)
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "execution_allowed": execution_allowed,
                "blocking_reason_count": len(blockers),
                "unknown_dimension_count": len(unknown_dimensions),
                "outputs": [str(md_path), str(json_path)],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
