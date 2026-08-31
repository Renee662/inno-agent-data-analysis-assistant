#!/usr/bin/env python3
"""Smoke-test workflow-fit states and the non-bypassable model gate."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "presets" / "data-analysis-assistant"
ASSESS = (
    PRESET
    / ".skills"
    / "run-statistical-analysis"
    / "scripts"
    / "assess_workflow_support.py"
)
APPROVE = (
    PRESET
    / ".skills"
    / "run-statistical-analysis"
    / "scripts"
    / "approve_model_spec.py"
)


def invoke(*args: str, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    if expect_success and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not expect_success and completed.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {' '.join(args)}")
    return completed


def assess(
    task: Path,
    output_dir: Path,
    *,
    observation: str = "independent",
    outcome_process: str = "standard",
    sampling: str = "simple",
    exposure: str = "not-applicable",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = output_dir / "workflow-decision-input.json"
    decision.write_text(
        json.dumps(
            {
                "observation_structure": observation,
                "outcome_process": outcome_process,
                "sampling_design": sampling,
                "count_exposure": exposure,
            }
        ),
        encoding="utf-8",
    )
    approval = output_dir / "workflow-approval.json"
    approval.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "approvalId": f"approval-{output_dir.name}",
                "status": "approved",
                "action": "approve-workflow-support",
                "source": "web-question-dialog",
                "questionId": f"question-{output_dir.name}",
                "sessionId": "smoke-session",
                "approvedAt": "2026-08-16T00:00:00+08:00",
                "artifact": {
                    "path": decision.name,
                    "sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    invoke(
        str(ASSESS),
        "--task",
        str(task),
        "--decision-input",
        str(decision),
        "--approval-record",
        str(approval),
        "--output-dir",
        str(output_dir),
    )
    return json.loads(
        (output_dir / "workflow-support-assessment.json").read_text(encoding="utf-8")
    )


def write_task(path: Path, goal: str = "association", decision_goal: str = "relationships") -> None:
    path.write_text(
        json.dumps(
            {
                "status": "approved",
                "approval": {"approval_id": "smoke-task"},
                "title": "Workflow gate smoke test",
                "goal": goal,
                "decision_goal": decision_goal,
                "outcome": "y",
                "predictors": ["x"],
                "controls": [],
                "variables": [
                    {"column": "y", "inferred_type": "numeric-continuous"},
                    {"column": "x", "inferred_type": "numeric-continuous"},
                ],
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="inno-workflow-support-") as temp_dir:
        root = Path(temp_dir)
        task = root / "task.json"
        write_task(task)

        supported = assess(task, root / "supported")
        assert supported["status"] == "supported"
        assert supported["execution_allowed"] is True

        unknown = assess(task, root / "unknown", observation="unknown")
        assert unknown["status"] == "needs-user-information"
        assert unknown["execution_allowed"] is False
        assert "记录之间的关系" in unknown["unknown_dimensions"]

        repeated = assess(task, root / "repeated", observation="repeated")
        assert repeated["status"] == "specialized-workflow-required"
        assert repeated["execution_allowed"] is False
        assert any(
            item["required_workflow"] == "混合效应模型或 GEE 工作流"
            for item in repeated["blocking_reasons"]
        )

        complex_case = assess(
            task,
            root / "complex",
            outcome_process="time-to-event",
            sampling="weighted",
            exposure="required",
        )
        assert complex_case["status"] == "specialized-workflow-required"
        assert len(complex_case["blocking_reasons"]) == 3

        prediction_task = root / "prediction-task.json"
        write_task(prediction_task, goal="prediction", decision_goal="prediction")
        prediction = assess(prediction_task, root / "prediction")
        assert prediction["status"] == "supported"
        assert prediction["execution_allowed"] is True
        assert prediction["blocking_reasons"] == []

        causal_task = root / "causal-task.json"
        write_task(causal_task, goal="causal")
        causal = assess(causal_task, root / "causal")
        assert causal["status"] == "specialized-workflow-required"
        assert any(
            item["code"] == "causal-identification-required"
            for item in causal["blocking_reasons"]
        )

        data = root / "cleaned-data.csv"
        data.write_text("x,y\n1,2\n2,4\n3,6\n4,8\n", encoding="utf-8")
        prep = root / "preparation-log.json"
        prep.write_text(
            json.dumps(
                {
                    "data_preparation_executed": True,
                    "modeling_executed": False,
                    "analysis_metadata": {
                        "analysis_exclusions": [],
                        "categorical_columns": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        blocked = invoke(
            str(APPROVE),
            "--task",
            str(task),
            "--workflow-support",
            str(root / "repeated" / "workflow-support-assessment.json"),
            "--preparation-log",
            str(prep),
            "--data",
            str(data),
            "--model-type",
            "ols",
            "--output-dir",
            str(root / "blocked-approval"),
            expect_success=False,
        )
        assert "not approved for modeling" in blocked.stderr
        assert not (root / "blocked-approval" / "approved-model-specification.json").exists()

        print(
            json.dumps(
                {
                    "ok": True,
                    "supported_status": supported["status"],
                    "unknown_status": unknown["status"],
                    "repeated_status": repeated["status"],
                    "prediction_status": prediction["status"],
                    "causal_status": causal["status"],
                    "blocked_approval_returncode": blocked.returncode,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
