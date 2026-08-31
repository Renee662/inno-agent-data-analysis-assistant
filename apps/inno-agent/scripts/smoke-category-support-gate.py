#!/usr/bin/env python3
"""Verify sparse-category evidence and separation approval blockers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    skill_root = app_root / "presets" / "data-analysis-assistant" / ".skills"
    support = load_module(
        "category_support_smoke", skill_root / "_shared" / "category_support.py"
    )

    clear = pd.DataFrame(
        {
            "workclass": ["A"] * 60 + ["B"] * 60,
            "income": (["no", "yes"] * 30) + (["no", "yes"] * 30),
        }
    )
    clear_screen = support.screen_category_support(
        clear,
        outcome="income",
        categorical_predictors={"workclass"},
        model_type="logistic",
        parameters_per_outcome_equation=3,
    )
    assert clear_screen["status"] == "clear"
    assert clear_screen["model_fitting_allowed"] is True

    sparse = pd.DataFrame(
        {
            "workclass": ["A"] * 60 + ["rare"] * 5,
            "income": (["no", "yes"] * 30) + ["no", "no", "no", "yes", "yes"],
        }
    )
    sparse_screen = support.screen_category_support(
        sparse,
        outcome="income",
        categorical_predictors={"workclass"},
        model_type="logistic",
        parameters_per_outcome_equation=3,
    )
    assert sparse_screen["status"] == "requires-review"
    assert sparse_screen["model_fitting_allowed"] is False
    rare = next(item for item in sparse_screen["factor_levels"] if item["level"] == "rare")
    assert "sparse-outcome-cell" in rare["risk_codes"]
    assert rare["blocking"] is False

    separated = pd.DataFrame(
        {
            "workclass": ["A"] * 60 + ["separated"] * 8,
            "income": (["no", "yes"] * 30) + ["yes"] * 8,
        }
    )
    separated_screen = support.screen_category_support(
        separated,
        outcome="income",
        categorical_predictors={"workclass"},
        model_type="logistic",
        parameters_per_outcome_equation=3,
    )
    assert separated_screen["status"] == "revision-required"
    assert separated_screen["model_fitting_allowed"] is False
    blocked = separated_screen["blocking_findings"][0]
    assert blocked["zero_outcome_categories"] == ["no"]
    assert "separation-candidate" in blocked["risk_codes"]

    model_scale_screen = support.screen_category_support(
        clear,
        outcome="income",
        categorical_predictors=set(),
        model_type="logistic",
        parameters_per_outcome_equation=10,
    )
    assert model_scale_screen["status"] == "requires-review"
    assert any(
        "observations_per_parameter" in item
        for item in model_scale_screen["review_findings"]
    )
    assert (
        model_scale_screen["planning_benchmarks"]["use_as_automatic_scientific_cutoff"]
        is False
    )

    design_points = pd.DataFrame(
        [(0, 0, "no"), (1, 0, "no"), (0, 1, "no"),
         (1, 1, "yes"), (2, 0, "yes"), (0, 2, "yes")] * 20,
        columns=["x1", "x2", "income"],
    )
    design_screen = support.screen_category_support(
        design_points,
        outcome="income",
        categorical_predictors=set(),
        model_type="logistic",
        parameters_per_outcome_equation=3,
        design_matrix=design_points[["x1", "x2"]],
        positive_class="yes",
    )
    assert design_screen["status"] == "revision-required"
    assert design_screen["design_separation"]["status"] == "complete-separation"
    assert any(
        item.get("reason") == "multivariable-complete-separation"
        for item in design_screen["blocking_findings"]
    )

    rows = support.flatten_category_support_rows(separated_screen)
    assert any(
        row["level"] == "separated"
        and row["outcome_count:no"] == 0
        and row["outcome_count:yes"] == 8
        for row in rows
    )

    approve_source = (
        skill_root / "run-statistical-analysis" / "scripts" / "approve_model_spec.py"
    ).read_text(encoding="utf-8")
    finalize_source = (
        skill_root / "run-statistical-analysis" / "scripts" / "finalize_model_spec.py"
    ).read_text(encoding="utf-8")
    run_source = (
        skill_root / "run-statistical-analysis" / "scripts" / "run_analysis.py"
    ).read_text(encoding="utf-8")
    report_source = (
        skill_root / "generate-final-report" / "scripts" / "generate_report.py"
    ).read_text(encoding="utf-8")
    skill_text = (skill_root / "run-statistical-analysis" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    for token in (
        "category-support-screen.csv",
        "flatten_category_support_rows",
        "category_support_screen",
    ):
        assert token in approve_source
    assert "unresolved sparse-category or separation risk" in finalize_source
    assert "runtime_category_support != spec.get" in run_source
    assert "每个类别的结果支持度（紧凑表）" in report_source
    assert "稀疏类别是某个类别样本或某种结果太少" in skill_text
    assert "Never merge, drop, relabel, or continue automatically" in skill_text

    print(
        json.dumps(
            {
                "ok": True,
                "clear": clear_screen["status"],
                "sparse": sparse_screen["status"],
                "separated": separated_screen["status"],
                "multivariable": design_screen["design_separation"]["status"],
                "rows": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
