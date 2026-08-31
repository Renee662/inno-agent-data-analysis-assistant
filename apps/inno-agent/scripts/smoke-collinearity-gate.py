#!/usr/bin/env python3
"""Verify redundant-variable screening and VIF interpretation gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import numpy as np


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
    approval = load_module(
        "approve_collinearity_smoke",
        skill_root
        / "run-statistical-analysis"
        / "scripts"
        / "approve_model_spec.py",
    )
    report = load_module(
        "report_collinearity_smoke",
        skill_root
        / "generate-final-report"
        / "scripts"
        / "generate_report.py",
    )

    frame = pd.DataFrame(
        {
            "education": ["HS", "Bachelors", "Masters", "HS", "Bachelors"],
            "education_num": [9, 13, 14, 9, 13],
            "age": [25, 35, 45, 55, 65],
        }
    )
    candidates = approval.screen_collinearity_candidates(
        frame,
        ["education", "education_num", "age"],
        {"education"},
    )
    assert any(
        {item["left"], item["right"]} == {"education", "education_num"}
        and item["reason"] == "deterministic-reencoding"
        for item in candidates
    )
    rank_deficient = approval.design_matrix_preflight(
        pd.DataFrame(
            {
                "age": [20.0, 30.0, 40.0, 50.0],
                "duplicated_age": [20.0, 30.0, 40.0, 50.0],
            }
        ),
        "logistic",
    )
    assert rank_deficient["status"] == "revision-required"
    assert rank_deficient["model_fitting_allowed"] is False
    assert rank_deficient["rank"] < rank_deficient["column_count"]
    assert "duplicated_age" in rank_deficient["dependent_columns"]
    full_rank = approval.design_matrix_preflight(
        pd.DataFrame(
            {
                "age": [20.0, 30.0, 40.0, 50.0],
                "score": [1.0, 4.0, 2.0, 8.0],
            }
        ),
        "logistic",
    )
    assert full_rank["status"] == "passed"
    assert np.isfinite(full_rank["rank"])

    spec = {
        "model_type": "logistic",
        "outcome": "income",
        "predictors": ["education", "education_num", "age"],
        "controls": [],
        "categorical_columns": ["education"],
        "categorical_reference_categories": {"education": "HS"},
        "positive_class": ">50K",
        "outcome_reference_class": "<=50K",
        "confidence_level": 0.95,
    }
    task = {
        "outcome": "income",
        "predictors": ["education", "education_num", "age"],
        "controls": [],
        "variable_metadata": {
            "income": {"display_name": "年收入是否超过5万美元"},
            "education": {"display_name": "教育程度"},
            "education_num": {"display_name": "教育年数"},
            "age": {"display_name": "年龄", "unit": "岁"},
        },
    }
    rows = [
        {
            "term": "education_Bachelors",
            "term_type": "coefficient",
            "estimate": "0.8",
            "std_error": "0.2",
            "ci_low": "0.4",
            "ci_high": "1.2",
            "p_value": "0.001",
            "p_value_adjusted_bh": "0.003",
            "factor_omnibus_p_value": "0.001",
            "factor_omnibus_p_value_adjusted_bh": "0.002",
            "multiplicity_supported": "True",
            "vif": "22.713",
            "collinearity_restricted": "True",
            "interpretation_supported": "False",
            "exp_estimate": "2.226",
        },
        {
            "term": "age",
            "term_type": "coefficient",
            "estimate": "0.04",
            "std_error": "0.01",
            "ci_low": "0.02",
            "ci_high": "0.06",
            "p_value": "0.002",
            "p_value_adjusted_bh": "0.004",
            "multiplicity_supported": "True",
            "vif": "1.2",
            "collinearity_restricted": "False",
            "interpretation_supported": "True",
            "exp_estimate": "1.041",
        },
    ]
    summary = {
        "model_type": "logistic",
        "metrics": {},
        "diagnostic_applicability": {
            "normal_qq": "not-applicable",
            "heteroskedasticity": "not-applicable",
        },
        "influence_available": False,
        "influential_cleaned_data_rows": [],
        "multiplicity": {
            "categorical_omnibus_tests": [
                {
                    "factor": "education",
                    "reference_category": "HS",
                    "degrees_of_freedom": 2,
                    "p_value": 0.001,
                    "p_value_adjusted_bh": 0.002,
                }
            ]
        },
        "collinearity": {
            "status": "severe",
            "maximum_vif": 22.713,
            "severe_terms": ["education_Bachelors", "education_num"],
            "severe_factors": ["education"],
        },
    }
    findings = report.build_plain_findings_html(rows, spec, task)
    assert "受严重共线性限制" in findings
    assert "方向较明确的正向关联主要见于年龄" in findings
    core = report.build_core_conclusion(spec, rows, task)
    assert "教育程度" in core and "未通过完整证据门槛" in core
    omnibus = report.build_factor_omnibus_html(summary, task, 0.05)
    assert "受严重共线性限制，先修订或做敏感性分析" in omnibus
    diagnostics = report.build_diagnostics_html(
        summary,
        [
            {"category": "multicollinearity", "metric": "VIF:education_Bachelors", "value": "22.713"},
            {"category": "multicollinearity", "metric": "VIF:age", "value": "1.2"},
        ],
        spec,
        task,
    )
    assert "不可作稳定独立作用解释" in diagnostics
    tests_html = report.build_statistical_tests_html(
        rows, "logistic", spec, summary, task
    )
    assert "不能据此稳定拆分它的独立作用" in tests_html

    run_source = (
        skill_root
        / "run-statistical-analysis"
        / "scripts"
        / "run_analysis.py"
    ).read_text(encoding="utf-8")
    for token in (
        "collinearity-review",
        "collinearity_restricted",
        "interpretation_supported",
        "VIF >= 10",
    ):
        assert token in run_source
    skill_text = (
        skill_root / "run-statistical-analysis" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "共线性是多个自变量包含大量重复信息" in skill_text
    assert "VIF越高，越难稳定拆分每个变量的独立作用" in skill_text
    finalize_source = (
        skill_root
        / "run-statistical-analysis"
        / "scripts"
        / "finalize_model_spec.py"
    ).read_text(encoding="utf-8")
    assert "deterministic duplicate encodings" in finalize_source
    print(json.dumps({"ok": True, "candidate_pairs": len(candidates), "design_preflight": full_rank["status"], "maximum_vif": 22.713}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
