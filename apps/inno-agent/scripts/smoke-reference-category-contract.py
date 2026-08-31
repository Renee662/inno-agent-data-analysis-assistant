#!/usr/bin/env python3
"""Verify categorical references control encoding and are disclosed consistently."""

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
    approval = load_module(
        "approve_model_spec_reference_smoke",
        skill_root
        / "run-statistical-analysis"
        / "scripts"
        / "approve_model_spec.py",
    )
    report = load_module(
        "generate_report_reference_smoke",
        skill_root
        / "generate-final-report"
        / "scripts"
        / "generate_report.py",
    )

    raw = pd.DataFrame(
        {
            "country": ["United-States", "China", "United-States", "Cuba"],
            "age": [30, 40, 50, 60],
        }
    )
    encoded = approval.encode_categorical_predictors(
        raw, {"country"}, {"country": "United-States"}
    )
    assert "country_United-States" not in encoded.columns
    assert {"country_China", "country_Cuba"}.issubset(encoded.columns)
    assert approval.parse_categorical_references(
        ["country=United-States"]
    ) == {"country": "United-States"}

    spec = {
        "model_type": "logistic",
        "outcome": "income",
        "predictors": ["country"],
        "controls": [],
        "categorical_columns": ["country"],
        "categorical_reference_categories": {"country": "United-States"},
        "positive_class": ">50K",
        "outcome_reference_class": "<=50K",
        "confidence_level": 0.95,
    }
    task = {
        "outcome": "income",
        "predictors": ["country"],
        "controls": [],
        "variable_metadata": {
            "income": {"display_name": "年收入"},
            # Deliberately wrong legacy metadata: approved spec must win.
            "country": {"display_name": "原籍国", "reference_category": "China"},
        },
    }
    row = {
        "term": "country_Cuba",
        "term_type": "coefficient",
        "estimate": "-0.7",
        "std_error": "0.2",
        "ci_low": "-1.1",
        "ci_high": "-0.3",
        "p_value": "0.01",
        "p_value_adjusted_bh": "0.02",
        "factor_omnibus_p_value": "0.001",
        "factor_omnibus_p_value_adjusted_bh": "0.001",
        "multiplicity_supported": "True",
        "exp_estimate": "0.497",
    }
    title, explanation, _status = report.effect_explanation(row, spec, task)
    assert "United-States" in title and "United-States" in explanation
    assert "China" not in title and "China" not in explanation

    method_html = report.build_method_html(
        spec,
        task,
        {"predictive_validation": {"status": "not-applicable"}},
    )
    assert "正类定义" in method_html and "&gt;50K" in method_html
    assert "二分类结果基准类别" in method_html and "&lt;=50K" in method_html
    assert "分类自变量参照组" in method_html and "United-States" in method_html
    results_html = report.build_results_html([row], "logistic", spec, task, compact=True)
    assert "自变量参照组" in results_html and "United-States" in results_html

    run_source = (
        skill_root
        / "run-statistical-analysis"
        / "scripts"
        / "run_analysis.py"
    ).read_text(encoding="utf-8")
    for contract_token in (
        "encode_categorical_predictors",
        "categorical_reference_categories",
        "predictor_reference_category",
        "outcome_reference_class",
    ):
        assert contract_token in run_source
    print(
        json.dumps(
            {
                "ok": True,
                "predictor_reference": "United-States",
                "positive_class": ">50K",
                "outcome_reference": "<=50K",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
