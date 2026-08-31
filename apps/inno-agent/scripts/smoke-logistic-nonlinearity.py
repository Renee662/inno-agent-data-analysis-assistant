#!/usr/bin/env python3
"""Verify Logistic functional-form approval, spline construction, and reporting gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
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
    approval = load_module(
        "approve_nonlinearity_smoke",
        skill_root / "run-statistical-analysis" / "scripts" / "approve_model_spec.py",
    )
    report = load_module(
        "report_nonlinearity_smoke",
        skill_root / "generate-final-report" / "scripts" / "generate_report.py",
    )

    age = np.arange(18, 78, dtype=float)
    frame = pd.DataFrame(
        {
            "age": age,
            "income": np.where((age < 28) | (age > 58), "low", "high"),
        }
    )
    forms = approval.parse_continuous_forms(["age=restricted-cubic-spline"])
    specifications = approval.continuous_form_specifications(
        frame, ["age"], forms, "income", "high"
    )
    age_spec = specifications["age"]
    assert age_spec["form"] == "restricted-cubic-spline"
    assert age_spec["term_names"] == ["age", "age__rcs1", "age__rcs2"]
    assert len(age_spec["knots_original_units"]) == 4
    assert len(age_spec["unadjusted_binned_rate_preview"]) >= 4
    transformed = approval.apply_continuous_forms(frame[["age"]], specifications)
    assert list(transformed.columns) == ["age", "age__rcs1", "age__rcs2"]
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()

    spec = {
        "model_type": "logistic",
        "outcome": "income",
        "predictors": ["age"],
        "controls": [],
        "categorical_columns": [],
        "categorical_reference_categories": {},
        "positive_class": "high",
        "outcome_reference_class": "low",
        "confidence_level": 0.95,
        "continuous_functional_forms": specifications,
    }
    task = {
        "outcome": "income",
        "predictors": ["age"],
        "controls": [],
        "variable_metadata": {
            "income": {"display_name": "年收入是否超过5万美元"},
            "age": {"display_name": "年龄", "unit": "岁"},
        },
    }
    rows = [
        {
            "term": "age",
            "term_type": "coefficient",
            "estimate": "0.1",
            "ci_low": "0.02",
            "ci_high": "0.18",
            "p_value": "0.01",
            "p_value_adjusted_bh": "0.02",
            "multiplicity_supported": "True",
            "collinearity_restricted": "False",
            "shape_basis_restricted": "False",
            "interpretation_supported": "True",
            "continuous_source_variable": "age",
            "continuous_functional_form": "restricted-cubic-spline",
            "nonlinear_basis_term": "False",
            "continuous_overall_p_value_adjusted_bh": "0.003",
        },
        {
            "term": "age__rcs1",
            "term_type": "coefficient",
            "estimate": "-0.4",
            "ci_low": "-0.7",
            "ci_high": "-0.1",
            "p_value": "0.02",
            "p_value_adjusted_bh": "0.03",
            "multiplicity_supported": "True",
            "collinearity_restricted": "False",
            "shape_basis_restricted": "True",
            "interpretation_supported": "False",
            "continuous_source_variable": "age",
            "continuous_functional_form": "restricted-cubic-spline",
            "nonlinear_basis_term": "True",
        },
    ]
    summary = {
        "continuous_shape_tests": [
            {
                "variable": "age",
                "form": "restricted-cubic-spline",
                "overall_p_value": 0.002,
                "overall_p_value_adjusted_bh": 0.003,
                "nonlinear_p_value": 0.01,
                "nonlinear_p_value_adjusted_bh": 0.01,
            }
        ],
        "multiplicity": {"categorical_omnibus_tests": []},
        "collinearity": {"status": "clear", "severe_factors": []},
        "metrics": {},
    }
    findings = report.build_plain_findings_html(rows, spec, task)
    assert "采用整体曲线解释" in findings
    assert "单个基函数系数不作为现实中的独立效应" in findings
    assert "age__rcs1" not in findings
    core = report.build_core_conclusion(spec, rows, task)
    assert "关系方向可能随取值改变" in core
    shape_html = report.build_continuous_shape_html(summary, task, 0.05)
    assert "限制性立方样条" in shape_html
    assert "非线性成分有证据" in shape_html

    run_source = (
        skill_root / "run-statistical-analysis" / "scripts" / "run_analysis.py"
    ).read_text(encoding="utf-8")
    for token in (
        "joint_wald_test",
        "continuous-shape-tests.csv",
        "adjusted-probability-",
        "shape_basis_restricted",
        "joint_shape_terms_excluded_from_individual_vif_gate",
    ):
        assert token in run_source
    skill_text = (skill_root / "run-statistical-analysis" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "determine continuous-variable forms automatically" in skill_text
    assert "do not ask them “linear, quadratic, or spline?”" in skill_text
    print(
        json.dumps(
            {
                "ok": True,
                "form": age_spec["form"],
                "basis_terms": age_spec["term_names"],
                "knots": age_spec["knots_original_units"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
