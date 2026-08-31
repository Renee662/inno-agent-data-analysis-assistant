#!/usr/bin/env python3
"""Verify that questionnaire receipts bind approval to exact artifact bytes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPROVE_TASK = (
    ROOT
    / "presets"
    / "data-analysis-assistant"
    / ".skills"
    / "plan-relationship-analysis"
    / "scripts"
    / "approve_analysis_task.py"
)


def run(*args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(str(value) for value in args)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


with tempfile.TemporaryDirectory(prefix="inno-human-approval-") as temporary:
    workspace = Path(temporary) / "workspace"
    proposal = workspace / "conversations" / "c1" / "work" / "analysis-plan" / "analysis-task.json"
    proposal.parent.mkdir(parents=True)
    proposal.write_text(
        json.dumps(
            {
                "status": "draft",
                "requires_user_confirmation": True,
                "dataset_summary": "测试数据",
                "research_question": "x 与 y 有什么关系？",
                "outcome": "y",
                "predictors": ["x"],
                "controls": [],
                "variable_metadata": {
                    "y": {"display_name": "结果", "user_confirmed": True},
                    "x": {"display_name": "因素", "user_confirmed": True},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    approval = workspace / ".approvals" / "session-1" / "approve-analysis-task-test.json"
    approval.parent.mkdir(parents=True)
    approval.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "approvalId": "approval-test",
                "status": "approved",
                "action": "approve-analysis-task",
                "source": "web-question-dialog",
                "questionId": "question-test",
                "sessionId": "session-1",
                "approvedAt": "2026-08-16T00:00:00+08:00",
                "artifact": {
                    "path": proposal.relative_to(workspace).as_posix(),
                    "sha256": hashlib.sha256(proposal.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    success = run(
        APPROVE_TASK,
        "--proposal",
        proposal,
        "--approval-record",
        approval,
        "--output-dir",
        proposal.parent,
    )
    if success.returncode != 0:
        raise AssertionError(success.stderr)
    approved = json.loads((proposal.parent / "approved-analysis-task.json").read_text(encoding="utf-8"))
    assert approved["status"] == "approved"
    assert approved["approval"]["question_id"] == "question-test"

    proposal.write_text(proposal.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = run(
        APPROVE_TASK,
        "--proposal",
        proposal,
        "--approval-record",
        approval,
        "--output-dir",
        proposal.parent / "changed",
    )
    assert changed.returncode != 0
    assert "changed after the user saw it" in changed.stderr
    print(json.dumps({"ok": True, "checks": 4}))
