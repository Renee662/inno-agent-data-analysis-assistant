#!/usr/bin/env python3
"""Promote an immutable analysis-task proposal using a questionnaire receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from approval import ApprovalError, approval_summary, read_object, verify_approval  # noqa: E402
from file_utils import atomic_replace_text  # noqa: E402


def fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(2)


def atomic_write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"Output already exists: {path}")
    atomic_replace_text(path, content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--approval-record", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_task(task: dict[str, Any]) -> None:
    if task.get("status") != "draft" or task.get("requires_user_confirmation") is not True:
        fail("Analysis-task proposal must still be an unapproved draft")
    for field in ("dataset_summary", "research_question", "outcome"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            fail(f"Analysis-task proposal is missing {field}")
    metadata = task.get("variable_metadata")
    if not isinstance(metadata, dict):
        fail("Analysis-task proposal has no variable_metadata")
    selected = [task.get("outcome"), *task.get("predictors", []), *task.get("controls", [])]
    unresolved = [
        str(column)
        for column in selected
        if not isinstance(metadata.get(str(column)), dict)
        or metadata[str(column)].get("user_confirmed") is not True
    ]
    if unresolved:
        fail("Selected variable meanings are not user-confirmed: " + ", ".join(unresolved))


def main() -> None:
    args = parse_args()
    proposal_path = args.proposal.resolve()
    approval_path = args.approval_record.resolve()
    try:
        task = read_object(proposal_path, "analysis-task proposal")
        approval = verify_approval(approval_path, "approve-analysis-task", proposal_path)
    except ApprovalError as exc:
        fail(str(exc))
    validate_task(task)
    approved = {
        **task,
        "status": "approved",
        "requires_user_confirmation": False,
        "approval": approval_summary(approval, approval_path),
        "proposal": str(proposal_path),
    }
    output_dir = args.output_dir.resolve()
    json_path = output_dir / "approved-analysis-task.json"
    md_path = output_dir / "approved-analysis-task.md"
    atomic_write(json_path, json.dumps(approved, ensure_ascii=False, indent=2) + "\n", args.overwrite)
    atomic_write(
        md_path,
        "\n".join(
            [
                "# 已批准的分析任务",
                "",
                f"- 研究问题：{approved['research_question']}",
                f"- 因变量：`{approved['outcome']}`",
                f"- 自变量：{', '.join(f'`{item}`' for item in approved.get('predictors', []))}",
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
