#!/usr/bin/env python3
"""End-to-end regression for non-standard table headers and shared read specs."""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
PRESET = APP_ROOT / "presets" / "data-analysis-assistant" / ".skills"
PROFILE = PRESET / "tabular-data-profiler" / "scripts" / "profile_data.py"
PLAN_ANALYSIS = PRESET / "plan-relationship-analysis" / "scripts" / "build_analysis_plan.py"
PLAN_PREP = PRESET / "plan-data-preparation" / "scripts" / "build_preparation_plan.py"
APPROVE_PREP = PRESET / "confirm-data-preparation" / "scripts" / "approve_preparation_plan.py"
EXECUTE_PREP = PRESET / "execute-data-preparation" / "scripts" / "execute_preparation.py"


def run(*args: object, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *(str(value) for value in args)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != expect:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expect}: {args}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with tempfile.TemporaryDirectory(prefix="inno-table-structure-") as temp_value:
    temp = Path(temp_value)

    standard_source = temp / "standard.csv"
    standard_source.write_text("age,outcome\n20,0\n30,1\n40,0\n", encoding="utf-8")
    standard_dir = temp / "standard-profile"
    run(PROFILE, standard_source, "--output-dir", standard_dir)
    standard_profile = read_json(standard_dir / "data-profile.json")
    standard_spec = standard_profile["table_read_spec"]["sheets"][0]
    assert standard_spec["status"] == "auto-confirmed"
    assert standard_spec["selected"]["header_rows"] == [1]
    assert standard_profile["profiles"][0]["row_count"] == 3

    sentinel_source = temp / "missing-sentinel.csv"
    sentinel_source.write_text(
        "workclass,income\nPrivate,0\n ?,1\n?  ,0\n",
        encoding="utf-8",
    )
    sentinel_dir = temp / "missing-sentinel-profile"
    run(PROFILE, sentinel_source, "--output-dir", sentinel_dir)
    sentinel_profile = read_json(sentinel_dir / "data-profile.json")
    sentinel_spec = sentinel_profile["table_read_spec"]["sheets"][0]["selected"]
    assert sentinel_spec["missing_value_tokens"] == ["?"]
    workclass = sentinel_profile["profiles"][0]["columns"][0]
    assert workclass["missing_count"] == 2
    assert workclass["non_null_count"] == 1
    execute_module = load_module(EXECUTE_PREP, "execute_preparation_smoke")
    sentinel_raw = execute_module.read_delimited(sentinel_source, None, header=None)
    sentinel_frame = execute_module.apply_table_read_spec(sentinel_raw, sentinel_spec)
    assert int(sentinel_frame["workclass"].isna().sum()) == 2

    headerless_source = temp / "headerless.csv"
    headerless_source.write_text("20,0\n30,1\n40,0\n", encoding="utf-8")
    headerless_dir = temp / "headerless-profile"
    run(PROFILE, headerless_source, "--output-dir", headerless_dir)
    headerless_profile = read_json(headerless_dir / "data-profile.json")
    headerless_spec = headerless_profile["table_read_spec"]["sheets"][0]
    assert headerless_spec["status"] == "pending-user-confirmation"
    assert headerless_spec["selected"]["header_rows"] == []
    run(
        PROFILE,
        headerless_source,
        "--headerless",
        "--structure-confirmed",
        "--output-dir",
        headerless_dir,
    )
    confirmed_headerless = read_json(headerless_dir / "data-profile.json")
    assert confirmed_headerless["table_read_spec"]["sheets"][0]["status"] == "user-confirmed"
    headerless_sensitive_review = confirmed_headerless["profiles"][0]["sensitive_review"]
    assert headerless_sensitive_review["status"] == "pending-semantic-review"
    assert headerless_sensitive_review["opaque_columns_requiring_semantic_review"] == [
        "column_1",
        "column_2",
    ]
    assert [item["column"] for item in confirmed_headerless["profiles"][0]["columns"]] == [
        "column_1",
        "column_2",
    ]

    excel_source = temp / "double-header.xlsx"
    pd.DataFrame(
        [
            ["X1", "X2", "Y"],
            ["ID", "LIMIT_BAL", "default payment next month"],
            [1, 20000, 1],
            [2, 120000, 0],
            [3, 90000, 0],
        ]
    ).to_excel(excel_source, sheet_name="Data", header=False, index=False)
    excel_dir = temp / "excel-profile"
    run(PROFILE, excel_source, "--output-dir", excel_dir)
    excel_profile = read_json(excel_dir / "data-profile.json")
    excel_spec = excel_profile["table_read_spec"]["sheets"][0]
    assert excel_spec["sheet"] == "Data"
    assert excel_spec["status"] == "pending-user-confirmation"
    assert excel_spec["selected"]["header_rows"] == [2]
    run(
        PROFILE,
        excel_source,
        "--sheet",
        "Data",
        "--header-rows",
        "2",
        "--structure-confirmed",
        "--output-dir",
        excel_dir,
    )
    confirmed_excel = read_json(excel_dir / "data-profile.json")
    assert confirmed_excel["profiles"][0]["row_count"] == 3
    assert [item["column"] for item in confirmed_excel["profiles"][0]["columns"]] == [
        "ID",
        "LIMIT_BAL",
        "default payment next month",
    ]

    source = temp / "double-header.csv"
    source.write_text(
        "X1,Y\n"
        "age,outcome\n"
        "20,0\n"
        "30,1\n"
        "40,0\n",
        encoding="utf-8",
    )

    profile_dir = temp / "profile"
    initial = run(PROFILE, source, "--output-dir", profile_dir)
    initial_result = json.loads(initial.stdout)
    assert initial_result["structure_confirmation_required"] is True
    initial_profile = read_json(profile_dir / "data-profile.json")
    sheet_spec = initial_profile["table_read_spec"]["sheets"][0]
    assert sheet_spec["status"] == "pending-user-confirmation"
    assert sheet_spec["selected"]["header_rows"] == [2]
    assert sheet_spec["selected"]["data_start_row"] == 3
    assert [item["column"] for item in initial_profile["profiles"][0]["columns"]] == [
        "age",
        "outcome",
    ]

    blocked = run(
        PLAN_ANALYSIS,
        "--profile",
        profile_dir / "data-profile.json",
        "--goal",
        "association",
        "--decision-goal",
        "relationships",
        "--outcome",
        "outcome",
        "--predictors",
        "age",
        "--output-dir",
        temp / "blocked-analysis",
        expect=2,
    )
    assert "Table structure confirmation required" in blocked.stderr

    run(
        PROFILE,
        source,
        "--header-rows",
        "2",
        "--structure-confirmed",
        "--output-dir",
        profile_dir,
    )
    confirmed_profile = read_json(profile_dir / "data-profile.json")
    confirmed_spec = confirmed_profile["table_read_spec"]["sheets"][0]
    assert confirmed_spec["status"] == "user-confirmed"
    assert confirmed_profile["profiles"][0]["row_count"] == 3
    assert not any(
        item["mixed_python_types"] for item in confirmed_profile["profiles"][0]["columns"]
    )

    analysis_dir = temp / "analysis"
    run(
        PLAN_ANALYSIS,
        "--profile",
        profile_dir / "data-profile.json",
        "--goal",
        "association",
        "--decision-goal",
        "relationships",
        "--outcome",
        "outcome",
        "--predictors",
        "age",
        "--output-dir",
        analysis_dir,
    )
    task = read_json(analysis_dir / "analysis-task.json")
    assert task["table_read_spec"]["header_rows"] == [2]
    task["status"] = "approved"
    task["requires_user_confirmation"] = False
    task["approval"] = {"approval_id": "table-structure-smoke"}
    approved_task = analysis_dir / "approved-analysis-task.json"
    approved_task.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")

    prep_dir = temp / "prep"
    run(
        PLAN_PREP,
        "--profile",
        profile_dir / "data-profile.json",
        "--task",
        approved_task,
        "--input",
        source,
        "--output-dir",
        prep_dir,
    )
    draft = read_json(prep_dir / "data-preparation-plan.json")
    assert draft["table_read_spec"] == task["table_read_spec"]

    decisions = temp / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "decisions": {
                    item["id"]: {"choice": item["recommendation"]}
                    for item in draft["pending_decisions"]
                    if item["recommendation"] in item["options"]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # This fixture has no discretionary decisions. Guard against a future rule
    # accidentally adding one without updating the explicit test contract.
    assert draft["pending_decisions"] == [], draft["pending_decisions"]

    approved_dir = temp / "approved"
    run(
        APPROVE_PREP,
        "--draft",
        prep_dir / "data-preparation-plan.json",
        "--decisions",
        decisions,
        "--output-dir",
        approved_dir,
    )
    approved = read_json(approved_dir / "approved-data-preparation-plan.json")
    assert approved["approved_decisions"] == []
    assert approved["approval"]["confirmation_method"] == "structured-item-questionnaire"
    assert approved["approval"]["whole_plan_reconfirmation_required"] is False

    choice_draft = json.loads(json.dumps(draft))
    choice_draft["pending_decisions"] = [
        {
            "id": "columns:unused",
            "topic": "无关字段",
            "status": "pending",
            "recommendation": "exclude",
            "options": ["exclude", "keep"],
        }
    ]
    choice_draft_path = temp / "choice-data-preparation-plan.json"
    choice_draft_path.write_text(
        json.dumps(choice_draft, ensure_ascii=False), encoding="utf-8"
    )
    choice_decisions = temp / "choice-user-decisions.json"
    choice_decisions.write_text(
        json.dumps(
            {"decisions": {"columns:unused": {"choice": "exclude"}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    choice_approved_dir = temp / "choice-approved"
    run(
        APPROVE_PREP,
        "--draft",
        choice_draft_path,
        "--decisions",
        choice_decisions,
        "--output-dir",
        choice_approved_dir,
    )
    choice_approved = read_json(
        choice_approved_dir / "approved-data-preparation-plan.json"
    )
    assert choice_approved["approved_decisions"][0]["selected_option"] == "exclude"
    assert choice_approved["approval"]["whole_plan_reconfirmation_required"] is False

    execution_dir = temp / "execution"
    run(
        EXECUTE_PREP,
        "--input",
        source,
        "--plan",
        approved_dir / "approved-data-preparation-plan.json",
        "--output-dir",
        execution_dir,
    )
    cleaned = pd.read_csv(execution_dir / "cleaned-data.csv")
    assert cleaned.columns.tolist() == ["age", "outcome"]
    assert len(cleaned) == 3

    print(
        json.dumps(
            {
                "ok": True,
                "standard_header": [1],
                "double_header": [2],
                "excel_double_header": [2],
                "headerless": True,
                "opaque_sensitive_review": "pending-semantic-review",
                "missing_sentinel_count": 2,
                "cleaned_rows": 3,
            }
        )
    )
