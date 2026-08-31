"""Verification helpers for server-issued human approval receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from file_utils import sha256_file


class ApprovalError(ValueError):
    """Raised when an approval receipt cannot authorize the requested action."""


def read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ApprovalError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ApprovalError(f"{label} must contain a JSON object")
    return value


def verify_approval(
    approval_path: Path,
    expected_action: str,
    artifact_path: Path,
) -> dict[str, Any]:
    record = read_object(approval_path, "approval record")
    if record.get("schemaVersion") != 1:
        raise ApprovalError("Approval record schemaVersion must be 1")
    if record.get("status") != "approved":
        raise ApprovalError("Approval record status is not approved")
    if record.get("action") != expected_action:
        raise ApprovalError(
            f"Approval action mismatch: expected {expected_action!r}, got {record.get('action')!r}"
        )
    if record.get("source") not in {"web-question-dialog", "tui-questionnaire"}:
        raise ApprovalError("Approval did not originate from a supported user questionnaire")
    for key in ("approvalId", "questionId", "sessionId", "approvedAt"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ApprovalError(f"Approval record is missing {key}")
    artifact = record.get("artifact")
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("sha256"), str)
        or not isinstance(artifact.get("path"), str)
    ):
        raise ApprovalError("Approval record has no artifact hash")
    resolved_approval = approval_path.resolve()
    workspace_root: Path | None = None
    for parent in resolved_approval.parents:
        if parent.name == ".approvals":
            workspace_root = parent.parent
            break
    if workspace_root is not None:
        try:
            expected_relative = artifact_path.resolve().relative_to(workspace_root).as_posix()
        except ValueError as exc:
            raise ApprovalError("Approved artifact is outside the approval workspace") from exc
        if artifact["path"].replace("\\", "/") != expected_relative:
            raise ApprovalError("Approval receipt refers to a different artifact path")
    elif Path(artifact["path"]).name != artifact_path.name:
        raise ApprovalError("Approval receipt refers to a different artifact name")
    current_hash = sha256_file(artifact_path)
    if artifact["sha256"] != current_hash:
        raise ApprovalError("Approved artifact changed after the user saw it")
    return record


def approval_summary(record: dict[str, Any], record_path: Path) -> dict[str, Any]:
    artifact = record.get("artifact", {})
    return {
        "approval_id": record.get("approvalId"),
        "question_id": record.get("questionId"),
        "session_id": record.get("sessionId"),
        "approved_at": record.get("approvedAt"),
        "source": record.get("source"),
        "approval_record": str(record_path),
        "approved_artifact_path": artifact.get("path"),
        "approved_artifact_sha256": artifact.get("sha256"),
    }
