#!/usr/bin/env python3
"""Smoke-test missingness evidence, stale-evidence refusal, and scope restriction."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PRESET = ROOT / "apps" / "inno-agent" / "presets" / "data-analysis-assistant"
PLANNER = PRESET / ".skills" / "plan-data-preparation" / "scripts" / "build_preparation_plan.py"
EXECUTOR = PRESET / ".skills" / "execute-data-preparation" / "scripts" / "execute_preparation.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, *args], text=True, capture_output=True, encoding="utf-8",
        env=environment,
    )
    if result.returncode != expect:
        raise AssertionError(
            f"Expected exit {expect}, got {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="missingness-gate-") as temp:
        root = Path(temp)
        source = root / "adult-mini.csv"
        source.write_text("income,x,group\n0,1,A\n1,?,A\n1,3,B\n0,4,B\n", encoding="utf-8")
        source_hash = sha256(source)
        selected = {
            "sheet": "CSV",
            "source_suffix": ".csv",
            "source_sha256": source_hash,
            "header_rows": [1],
            "data_start_row": 2,
            "column_names": ["income", "x", "group"],
            "missing_value_tokens": ["?"],
        }
        profile = {
            "source_file": source.name,
            "table_read_spec": {"sheets": [{"sheet": "CSV", "status": "auto-confirmed", "selected": selected}]},
            "profiles": [
                {
                    "sheet": "CSV",
                    "duplicate_row_count": 0,
                    "fully_empty_row_count": 0,
                    "columns": [
                        {"column": "income", "inferred_type": "boolean", "missing_rate": 0, "unique_count": 2},
                        {"column": "x", "inferred_type": "numeric-continuous", "missing_rate": 0.25, "unique_count": 3},
                        {"column": "group", "inferred_type": "categorical", "missing_rate": 0, "unique_count": 2},
                    ],
                }
            ],
        }
        task = {
            "status": "approved",
            "requires_user_confirmation": False,
            "approval": {"source": "smoke"},
            "sheet": "CSV",
            "table_read_spec": selected,
            "goal": "association",
            "unit_of_analysis": "person",
            "outcome": "income",
            "predictors": ["x", "group"],
            "controls": [],
        }
        profile_path = root / "profile.json"
        task_path = root / "task.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        task_path.write_text(json.dumps(task), encoding="utf-8")
        planned = root / "planned"
        run(
            str(PLANNER), "--input", str(source), "--profile", str(profile_path),
            "--task", str(task_path), "--output-dir", str(planned),
        )
        draft_path = planned / "data-preparation-plan.json"
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        screen = draft["missingness_bias_screen"]
        assert screen["complete_case_rows"] == 3
        assert screen["complete_case_excluded_rows"] == 1
        assert screen["interpretation"]["can_prove_no_selection_bias"] is False
        assert draft["missingness_conclusion_contract"]["scope"] == "analyzed-sample-only"

        approved = json.loads(json.dumps(draft))
        approved["status"] = "approved"
        approved["requires_user_confirmation"] = False
        approved["data_preparation_executed"] = False
        approved["pending_decisions"] = []
        decision = next(item for item in draft["pending_decisions"] if item["id"] == "missing:x")
        decision["status"] = "user-confirmed"
        decision["selected_option"] = "median-imputation"
        approved["approved_decisions"] = [decision]
        approved["approval"] = {
            "source_plan": str(draft_path),
            "source_plan_sha256": sha256(draft_path),
        }
        approved_path = root / "approved.json"
        approved_path.write_text(json.dumps(approved), encoding="utf-8")
        executed = root / "executed"
        run(str(EXECUTOR), "--input", str(source), "--plan", str(approved_path), "--output-dir", str(executed))
        log = json.loads((executed / "data-preparation-log.json").read_text(encoding="utf-8"))
        assert log["post_preparation_missingness"]["rows_with_any_selected_missingness"] == 0
        assert log["missingness_bias_screen"] == screen
        assert (executed / "missingness-impact.csv").is_file()

        stale = json.loads(json.dumps(approved))
        stale["missingness_bias_screen"]["complete_case_excluded_rows"] = 0
        stale_path = root / "stale-approved.json"
        stale_path.write_text(json.dumps(stale), encoding="utf-8")
        refused = run(
            str(EXECUTOR), "--input", str(source), "--plan", str(stale_path),
            "--output-dir", str(root / "refused"), expect=2,
        )
        assert "differs from the approved plan" in refused.stderr
    print("missingness bias gate smoke test passed")


if __name__ == "__main__":
    main()
