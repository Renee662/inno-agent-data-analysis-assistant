#!/usr/bin/env python3
"""Finalize a model proposal only with a matching questionnaire receipt."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from approval import ApprovalError, approval_summary, read_object, sha256_file, verify_approval  # noqa: E402
from file_utils import atomic_replace_text  # noqa: E402

CONTINUOUS_FORM_SOURCE_LABELS = {
    "explicit-approved-domain-override": "已批准的专业设定",
    "approved-task-metadata": "已批准任务中的专业设定",
    "automatic-fixed-flexible-default": "系统固定的灵活默认规则",
    "automatic-limited-support-linear": "取值支持不足时的保守直线规则",
}


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--approval-record", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--workflow-support", required=True, type=Path)
    parser.add_argument("--preparation-log", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"Output already exists: {path}")
    atomic_replace_text(path, content)


def main() -> None:
    args = parse_args()
    proposal_path = args.proposal.resolve()
    approval_path = args.approval_record.resolve()
    try:
        proposal = read_object(proposal_path, "model-specification proposal")
        approval = verify_approval(
            approval_path, "approve-model-specification", proposal_path
        )
    except ApprovalError as exc:
        fail(str(exc))
    if proposal.get("status") != "draft" or proposal.get("requires_user_confirmation") is not True:
        fail("Model proposal is not awaiting approval")
    design_preflight = proposal.get("design_matrix_preflight", {})
    if (
        not isinstance(design_preflight, dict)
        or design_preflight.get("status") != "passed"
        or design_preflight.get("model_fitting_allowed") is not True
    ):
        fail("Model proposal has not passed the final design-matrix preflight")
    collinearity_screen = proposal.get("collinearity_screen", {})
    if (
        isinstance(collinearity_screen, dict)
        and collinearity_screen.get("model_fitting_allowed") is False
    ):
        fail(
            "Model proposal contains deterministic duplicate encodings; revise the variable set before approval"
        )
    category_support_screen = proposal.get("category_support_screen", {})
    if (
        isinstance(category_support_screen, dict)
        and category_support_screen.get("model_fitting_allowed") is False
    ):
        fail(
            "Model proposal contains unresolved sparse-category or separation risk; revise category handling, "
            "reduce the approved model, provide more data, or stop before approval"
        )
    iia_check = proposal.get("iia_check", {})
    if proposal.get("model_type") == "multinomial-logistic" and (
        not isinstance(iia_check, dict)
        or iia_check.get("model_fitting_allowed") is not True
    ):
        fail(
            "Ordinary multinomial logistic cannot be approved because IIA sensitivity "
            "was detected or not evaluated; use a specialized nested/multinomial-probit "
            "choice workflow or revise the outcome structure"
        )
    proportional_odds_check = proposal.get("proportional_odds_check", {})
    if proposal.get("model_type") == "ordinal-logistic" and (
        not isinstance(proportional_odds_check, dict)
        or proportional_odds_check.get("model_fitting_allowed") is not True
    ):
        fail(
            "Ordinary ordinal logistic cannot be approved because the proportional-odds assumption "
            "was violated or not evaluated; use multinomial logistic or a specialized partial-proportional-odds workflow"
        )
    count_dispersion_check = proposal.get("count_dispersion_check", {})
    if proposal.get("model_type") == "poisson" and (
        not isinstance(count_dispersion_check, dict)
        or count_dispersion_check.get("model_fitting_allowed") is not True
    ):
        fail(
            "Ordinary Poisson cannot be approved because adjusted overdispersion was detected "
            "or not evaluated; return to model choice for negative-binomial regression or stop"
        )
    negative_binomial_need_check = proposal.get("negative_binomial_need_check", {})
    if proposal.get("model_type") == "negative-binomial" and (
        not isinstance(negative_binomial_need_check, dict)
        or negative_binomial_need_check.get("model_fitting_allowed") is not True
    ):
        fail(
            "Negative-binomial regression cannot be approved because adjusted extra dispersion "
            "was not detected or not evaluated; return to model choice for Poisson or stop"
        )
    zero_inflation_check = proposal.get("zero_inflation_check", {})
    if proposal.get("model_type") in {"poisson", "negative-binomial"} and (
        not isinstance(zero_inflation_check, dict)
        or zero_inflation_check.get("model_fitting_allowed") is not True
    ):
        fail(
            "Ordinary count regression cannot be approved because excess zeros were detected "
            "or not evaluated; use a specialized hurdle/zero-inflated workflow or stop"
        )

    task_path = args.task.resolve()
    workflow_path = args.workflow_support.resolve()
    prep_path = args.preparation_log.resolve()
    data_path = args.data.resolve()
    task = read_object(task_path, "approved analysis task")
    workflow = read_object(workflow_path, "workflow assessment")
    if task.get("status") != "approved" or not isinstance(task.get("approval"), dict):
        fail("Analysis task is not backed by a questionnaire approval")
    if workflow.get("status") != "supported" or not isinstance(workflow.get("approval"), dict):
        fail("Workflow assessment is not backed by a questionnaire approval")

    provenance = proposal.get("provenance")
    if not isinstance(provenance, dict):
        fail("Model proposal has no provenance hashes")
    expected = {
        "analysis_task_sha256": sha256_file(task_path),
        "workflow_support_assessment_sha256": sha256_file(workflow_path),
        "preparation_log_sha256": sha256_file(prep_path),
        "cleaned_data_sha256": sha256_file(data_path),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            fail(f"Model proposal input changed after proposal generation: {key}")

    approved = {
        **proposal,
        "status": "approved",
        "requires_user_confirmation": False,
        "user_confirmed": True,
        "approved_at": datetime.now().astimezone().isoformat(),
        "approval": approval_summary(approval, approval_path),
        "proposal": str(proposal_path),
        "proposal_sha256": sha256_file(proposal_path),
    }
    output_dir = args.output_dir.resolve()
    json_path = output_dir / "approved-model-specification.json"
    md_path = output_dir / "approved-model-specification.md"
    atomic_write(json_path, json.dumps(approved, ensure_ascii=False, indent=2) + "\n", args.overwrite)
    atomic_write(
        md_path,
        "\n".join(
            [
                "# 已批准的模型规格",
                "",
                f"- 模型：`{approved.get('model_type')}`",
                f"- 因变量：`{approved.get('outcome')}`",
                f"- 自变量：{', '.join(f'`{item}`' for item in approved.get('predictors', []))}",
                f"- 控制变量：{', '.join(f'`{item}`' for item in approved.get('controls', [])) or '无'}",
                *(
                    [
                        f"- 二分类目标事件：`{approved.get('positive_class')}`",
                        f"- 二分类结果基准类别：`{approved.get('outcome_reference_class')}`",
                    ]
                    if approved.get("model_type") == "logistic"
                    else []
                ),
                *(
                    [f"- 多分类结果参照类别：`{approved.get('reference_class')}`"]
                    if approved.get("model_type") == "multinomial-logistic"
                    else []
                ),
                *(
                    [
                        f"- IIA假设检查：`{approved.get('iia_check', {}).get('status', 'not-evaluated')}`",
                        f"  - 最小Holm校正p值：{approved.get('iia_check', {}).get('minimum_adjusted_p_value', '未得到')}",
                    ]
                    if approved.get("model_type") == "multinomial-logistic"
                    else []
                ),
                *(
                    ["- 分类自变量参照组："]
                    + [
                        f"  - `{column}`：`{reference}`"
                        for column, reference in sorted(
                            approved.get("categorical_reference_categories", {}).items()
                        )
                    ]
                    if isinstance(approved.get("categorical_reference_categories"), dict)
                    and approved.get("categorical_reference_categories")
                    else []
                ),
                *(
                    ["- 连续变量函数形式："]
                    + [
                        f"  - `{column}`：{item.get('plain_label', item.get('form'))}；来源={CONTINUOUS_FORM_SOURCE_LABELS.get(str(item.get('selection_source')), item.get('selection_source', '未记录'))}；理由={item.get('selection_rationale', '未记录')}"
                        for column, item in sorted(
                            approved.get("continuous_functional_forms", {}).items()
                        )
                        if isinstance(item, dict)
                    ]
                    if isinstance(approved.get("continuous_functional_forms"), dict)
                    and approved.get("continuous_functional_forms")
                    else []
                ),
                f"- 类别支持度检查：`{approved.get('category_support_screen', {}).get('status', 'not-applicable')}`",
                *(
                    [
                        f"- 比例优势假设检查：`{approved.get('proportional_odds_check', {}).get('status', 'not-evaluated')}`",
                        f"  - p值：{approved.get('proportional_odds_check', {}).get('p_value', '未得到')}",
                    ]
                    if approved.get("model_type") == "ordinal-logistic"
                    else []
                ),
                *(
                    [
                        f"- Poisson过度离散检查：`{approved.get('count_dispersion_check', {}).get('status', 'not-evaluated')}`",
                        f"  - 单侧p值：{approved.get('count_dispersion_check', {}).get('p_value', '未得到')}；Pearson离散度（仅描述）：{approved.get('count_dispersion_check', {}).get('pearson_dispersion_descriptive', '未得到')}",
                    ]
                    if approved.get("model_type") == "poisson"
                    else []
                ),
                *(
                    [
                        f"- 负二项必要性检查：`{approved.get('negative_binomial_need_check', {}).get('status', 'not-evaluated')}`",
                        f"  - 单侧p值：{approved.get('negative_binomial_need_check', {}).get('p_value', '未得到')}",
                    ]
                    if approved.get("model_type") == "negative-binomial"
                    else []
                ),
                *(
                    [
                        f"- 过多零值检查：`{approved.get('zero_inflation_check', {}).get('status', 'not-evaluated')}`",
                        f"  - 实际零值：{approved.get('zero_inflation_check', {}).get('observed_zero_count', '未得到')}；模型预计零值：{approved.get('zero_inflation_check', {}).get('expected_zero_count', '未得到')}",
                    ]
                    if approved.get("model_type") in {"poisson", "negative-binomial"}
                    else []
                ),
                f"- 置信水平：{float(approved.get('confidence_level', 0.95)):.1%}",
                f"- 审批编号：`{approval.get('approvalId')}`",
                f"- 问卷编号：`{approval.get('questionId')}`",
                f"- 已批准提案 SHA-256：`{approval['artifact']['sha256']}`",
                "",
            ]
        ),
        args.overwrite,
    )
    print(json.dumps({"ok": True, "status": "approved", "outputs": [str(json_path), str(md_path)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
