#!/usr/bin/env python3
"""Smoke-test that semantic review asks only about analysis-relevant fields."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    preset_root = app_root / "presets" / "data-analysis-assistant"
    profiler = (
        preset_root
        / ".skills"
        / "tabular-data-profiler"
        / "scripts"
        / "profile_data.py"
    )
    discovery = (
        preset_root
        / ".skills"
        / "tabular-data-profiler"
        / "scripts"
        / "discover_context.py"
    )
    planner = (
        preset_root
        / ".skills"
        / "plan-relationship-analysis"
        / "scripts"
        / "build_analysis_plan.py"
    )
    report_generator = (
        preset_root
        / ".skills"
        / "generate-final-report"
        / "scripts"
        / "generate_report.py"
    )

    with tempfile.TemporaryDirectory(prefix="inno-semantic-review-") as temp_dir:
        fixture = Path(temp_dir)
        table = fixture / "speed_dating.csv"
        profile_dir = fixture / "profile"
        plan_dir = fixture / "plan"
        row_count = 60
        data: dict[str, list[int | float]] = {
            "iid": [index // 3 + 1 for index in range(row_count)],
            "pid": [(index * 7) % 20 + 1 for index in range(row_count)],
            "batch_id": [index // 10 + 1 for index in range(row_count)],
            "match": [index % 2 for index in range(row_count)],
            "attr": [round(4.0 + (index % 7) * 0.5, 1) for index in range(row_count)],
            "sinc": [round(3.0 + (index % 5) * 0.75, 2) for index in range(row_count)],
            "wave": [index // 12 + 1 for index in range(row_count)],
        }
        for index in range(93):
            data[f"noise_{index:03d}"] = [
                (row + index) % 11 for row in range(row_count)
            ]
        pd.DataFrame(data).to_csv(table, index=False)
        (fixture / "README.md").write_text(
            "# Speed Dating study codebook\n"
            "match: whether both participants wanted a second date (0=no, 1=yes).\n"
            "attr: attractiveness rating from 1 to 10.\n"
            "sinc: sincerity rating from 1 to 10.\n"
            "wave: event wave in which the rating was recorded.\n"
            "iid: participant identifier. pid: rated partner identifier.\n",
            encoding="utf-8",
        )

        run([sys.executable, str(profiler), str(table), "--output-dir", str(profile_dir)])
        run(
            [
                sys.executable,
                str(discovery),
                "--table",
                str(table),
                "--profile",
                str(profile_dir / "data-profile.json"),
                "--context-root",
                str(fixture),
                "--output-dir",
                str(profile_dir),
            ]
        )
        run(
            [
                sys.executable,
                str(planner),
                "--profile",
                str(profile_dir / "data-profile.json"),
                "--context",
                str(profile_dir / "dataset-context.json"),
                "--goal",
                "association",
                "--decision-goal",
                "relationships",
                "--outcome",
                "match",
                "--predictors",
                "attr",
                "sinc",
                "--controls",
                "wave",
                "--output-dir",
                str(plan_dir),
            ]
        )

        task = json.loads((plan_dir / "analysis-task.json").read_text(encoding="utf-8"))
        semantic = task["semantic_review"]
        selected = [item["column"] for item in semantic["fields"]]
        assert selected == ["match", "attr", "sinc", "wave"], selected
        assert semantic["question_batches"] == [["match", "attr", "sinc"], ["wave"]]
        assert all(len(batch) <= 3 for batch in semantic["question_batches"])
        assert all(item["source_evidence"] for item in semantic["fields"])
        assert all(
            item["interaction_template"]["help_or_objection_is_confirmation"] is False
            for item in semantic["fields"]
        )
        assert not any(name.startswith("noise_") for name in selected)

        structural = [item["column"] for item in semantic["structural_candidates"]]
        assert len(structural) <= 3
        assert {"iid", "pid"}.issubset(set(structural)), structural
        assert not any(name.startswith("noise_") for name in structural)

        context = json.loads(
            (profile_dir / "dataset-context.json").read_text(encoding="utf-8")
        )
        assert context["network_used"] is False

        report_module = load_module(report_generator, "generate_report_sensitive_smoke")
        opaque_profile = {
            "profiles": [
                {
                    "sheet": "CSV",
                    "sensitive_review": {
                        "opaque_columns_requiring_semantic_review": [
                            "column_1",
                            "column_9",
                            "column_10",
                        ]
                    },
                    "columns": [
                        {"column": "column_1", "sensitive_name_candidate": False},
                        {"column": "column_9", "sensitive_name_candidate": False},
                        {"column": "column_10", "sensitive_name_candidate": False},
                    ],
                }
            ]
        }
        confirmed_task = {
            "variable_metadata": {
                "column_9": {
                    "display_name": "种族",
                    "meaning": "人口普查记录中的种族类别",
                    "user_confirmed": True,
                },
                "column_10": {
                    "display_name": "性别",
                    "meaning": "受访者性别",
                    "user_confirmed": True,
                },
            }
        }
        sensitive, unresolved = report_module.sensitive_field_review(
            opaque_profile, confirmed_task
        )
        assert sensitive == ["种族（column_9）", "性别（column_10）"], sensitive
        assert unresolved == ["column_1"], unresolved
        quality_html = report_module.build_quality_html(
            opaque_profile, {"actions": [], "deferred_actions": []}, confirmed_task
        )
        assert "种族（column_9）" in quality_html
        assert "待核对（1）" in quality_html
        assert "不能据此断言它们不敏感" in quality_html

        roles = report_module.eda_variable_roles(
            {
                "variables": [
                    {"column": "income_code", "inferred_type": "numeric"},
                    {"column": "age", "inferred_type": "numeric"},
                    {"column": "workclass_code", "inferred_type": "numeric"},
                ],
                "variable_metadata": {
                    "workclass_code": {
                        "category_meanings": {"1": "私营", "2": "政府"}
                    }
                },
            },
            {
                "outcome": "income_code",
                "model_type": "logistic",
                "categorical_columns": ["workclass_code"],
            },
            ["income_code", "age", "workclass_code"],
        )
        assert roles == {
            "income_code": "binary",
            "age": "continuous",
            "workclass_code": "nominal",
        }, roles
        multinomial_roles = report_module.eda_variable_roles(
            {},
            {"outcome": "class_code", "model_type": "multinomial-logistic"},
            ["class_code"],
        )
        assert multinomial_roles["class_code"] == "nominal"
        ordinal_roles = report_module.eda_variable_roles(
            {},
            {"outcome": "level_code", "model_type": "ordinal-logistic"},
            ["level_code"],
        )
        assert ordinal_roles["level_code"] == "ordinal"

        print(
            json.dumps(
                {
                    "ok": True,
                    "total_columns": len(data),
                    "semantic_fields": selected,
                    "question_batches": semantic["question_batches"],
                    "structural_candidates": structural,
                    "network_used": context["network_used"],
                    "semantic_sensitive_fields": sensitive,
                    "unresolved_sensitive_fields": unresolved,
                    "eda_roles": roles,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
