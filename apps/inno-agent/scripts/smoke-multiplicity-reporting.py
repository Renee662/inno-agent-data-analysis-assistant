#!/usr/bin/env python3
"""Verify hierarchical multiplicity decisions control report highlights."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_report_multiplicity_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    skill_root = app_root / "presets" / "data-analysis-assistant" / ".skills"
    report = load_module(
        skill_root
        / "generate-final-report"
        / "scripts"
        / "generate_report.py"
    )
    spec = {
        "model_type": "logistic",
        "outcome": "income",
        "predictors": ["country", "age"],
        "controls": [],
        "categorical_columns": ["country"],
        "categorical_reference_categories": {"country": "United-States"},
        "positive_class": ">50K",
        "outcome_reference_class": "<=50K",
        "confidence_level": 0.95,
    }
    task = {
        "outcome": "income",
        "predictors": ["country", "age"],
        "controls": [],
        "variable_metadata": {
            "income": {"display_name": "年收入是否超过5万美元"},
            "country": {"display_name": "原籍国"},
            "age": {"display_name": "年龄", "unit": "岁"},
        },
    }
    rows = [
        {
            "term": "country_China",
            "term_type": "coefficient",
            "estimate": "-1.0",
            "std_error": "0.2",
            "ci_low": "-1.4",
            "ci_high": "-0.6",
            "p_value": "0.001",
            "p_value_adjusted_bh": "0.003",
            "factor_omnibus_p_value": "0.04",
            "factor_omnibus_p_value_adjusted_bh": "0.08",
            "multiplicity_supported": "False",
            "exp_estimate": "0.368",
        },
        {
            "term": "age",
            "term_type": "coefficient",
            "estimate": "0.05",
            "std_error": "0.01",
            "ci_low": "0.03",
            "ci_high": "0.07",
            "p_value": "0.005",
            "p_value_adjusted_bh": "0.01",
            "multiplicity_supported": "True",
            "exp_estimate": "1.051",
        },
    ]
    summary = {
        "metrics": {"llr_p_value": 0.001},
        "multiplicity": {
            "coefficient_method": "Benjamini-Hochberg FDR",
            "categorical_omnibus_tests": [
                {
                    "factor": "country",
                    "reference_category": "United-States",
                    "degrees_of_freedom": 2,
                    "p_value": 0.04,
                    "p_value_adjusted_bh": 0.08,
                }
            ],
        },
    }

    findings = report.build_plain_findings_html(rows, spec, task)
    assert "年龄" in findings and "方向较明确的正向关联" in findings
    assert "原籍国：China 与 United-States" in findings
    assert "另有 1 项未同时通过" in findings
    assert "多重比较校正后证据不足" in findings
    core = report.build_core_conclusion(spec, rows, task)
    assert "年龄与年收入是否超过5万美元呈正向条件关联" in core
    assert "原籍国未通过完整证据门槛" in core

    omnibus = report.build_factor_omnibus_html(summary, task, 0.05)
    assert "原始p值" in omnibus and "BH校正p值" in omnibus
    assert "不展开具体类别强结论" in omnibus

    tests = report.build_statistical_tests_html(
        rows, "logistic", spec, summary, task
    )
    assert "BH校正且未受严重共线性限制的项目共 1 项" in tests
    assert "BH校正p值为" in tests

    run_source = (
        skill_root
        / "run-statistical-analysis"
        / "scripts"
        / "run_analysis.py"
    ).read_text(encoding="utf-8")
    for token in (
        "def benjamini_hochberg",
        "def categorical_omnibus_tests",
        "factor-omnibus-tests.csv",
        "multiplicity_supported",
    ):
        assert token in run_source
    run_tree = ast.parse(run_source)
    selected = [
        node
        for node in run_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"benjamini_hochberg", "categorical_omnibus_tests"}
    ]

    def fail(message: str) -> None:
        raise AssertionError(message)

    extracted: dict[str, Any] = {"np": np, "stats": stats, "Any": Any, "fail": fail}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "run-analysis-extract", "exec"), extracted)
    adjusted = extracted["benjamini_hochberg"]([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])

    class FakeResult:
        params = np.array([0.0, 1.0, 2.0, 0.1])

        @staticmethod
        def cov_params():
            return np.eye(4)

    omnibus_values = extracted["categorical_omnibus_tests"](
        FakeResult(),
        "logistic",
        ["const", "country_China", "country_Cuba", "age"],
        {"country": "United-States"},
    )
    assert len(omnibus_values) == 1
    assert omnibus_values[0]["degrees_of_freedom"] == 2
    assert np.isclose(omnibus_values[0]["statistic"], 5.0)
    print(json.dumps({"ok": True, "highlighted": 1, "blocked_by_omnibus": 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
