#!/usr/bin/env python3
"""Verify that large categorical results stay concise and cautiously worded."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_report_compact_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    report = load_module(
        app_root
        / "presets"
        / "data-analysis-assistant"
        / ".skills"
        / "generate-final-report"
        / "scripts"
        / "generate_report.py"
    )
    template = (
        app_root
        / "presets"
        / "data-analysis-assistant"
        / ".skills"
        / "generate-final-report"
        / "assets"
        / "report-template.html"
    ).read_text(encoding="utf-8")
    spec = {
        "model_type": "logistic",
        "outcome": "income_high",
        "predictors": ["country"],
        "controls": [],
        "categorical_columns": ["country"],
        "categorical_reference_categories": {"country": "United-States"},
        "positive_class": ">50K",
        "outcome_reference_class": "<=50K",
        "confidence_level": 0.95,
    }
    task = {
        "outcome": "income_high",
        "predictors": ["country"],
        "controls": [],
        "variable_metadata": {
            "income_high": {"display_name": "年收入是否超过5万美元"},
            "country": {"display_name": "原籍国", "reference_category": "United-States"},
        },
    }
    rows = []
    for index in range(1, 13):
        estimate = -0.7 if index <= 4 else (0.55 if index <= 8 else 0.08)
        low, high = ((-1.1, -0.3) if index <= 4 else ((0.2, 0.9) if index <= 8 else (-0.4, 0.5)))
        rows.append(
            {
                "term": f"country_Level{index:02d}",
                "term_type": "coefficient",
                "estimate": str(estimate),
                "std_error": "0.2",
                "ci_low": str(low),
                "ci_high": str(high),
                "p_value": "0.01" if index <= 8 else "0.7",
                "p_value_adjusted_bh": "0.02" if index <= 8 else "0.7",
                "factor_omnibus_p_value": "0.001",
                "factor_omnibus_p_value_adjusted_bh": "0.001",
                "multiplicity_supported": "True" if index <= 8 else "False",
                "exp_estimate": str(math.exp(estimate)),
            }
        )

    findings = report.build_plain_findings_html(rows, spec, task)
    assert "finding-card" not in findings
    assert findings.count("<tr>") == 7  # one header row plus six key results
    assert "另有 4 项" in findings
    assert "完整参数结果以紧凑表格" in findings
    assert "在其他已纳入因素相同的情况下" not in findings

    uncertain_title, uncertain_text, uncertain_status = report.effect_explanation(
        rows[-1], spec, task
    )
    assert "Level12" in uncertain_title
    assert "包含1" in uncertain_text
    assert "相对更高" not in uncertain_text and "相对更低" not in uncertain_text
    assert "不确定" in uncertain_status

    tests_html = report.build_statistical_tests_html(
        rows,
        "logistic",
        spec,
        {"metrics": {"llr_p_value": 0.01}},
        task,
    )
    assert "完整参数表已移至" in tests_html
    assert "compact-results" not in tests_html
    assert tests_html.count("现有样本不足以排除") == 0

    appendix_table = report.build_results_html(
        rows, "logistic", spec, task, compact=True
    )
    assert "compact-results" in appendix_table
    assert appendix_table.count("country_Level") == len(rows)
    appendix = report.build_diagnostic_appendix_html(
        {"profiles": [{"columns": []}]},
        {"actions": [], "deferred_actions": []},
        {"influential_cleaned_data_rows": []},
        [],
        rows,
        spec,
        task,
    )
    assert "完整模型参数（紧凑表）" in appendix
    assert appendix.count("country_Level") == len(rows)
    assert ".compact-results" in template
    print(json.dumps({"ok": True, "result_rows": len(rows), "main_rows": 6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
