#!/usr/bin/env python3
"""Verify OLS nonlinear-form approval, fitting, and report wording."""

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
        "approve_ols_nonlinearity_smoke",
        skill_root / "run-statistical-analysis" / "scripts" / "approve_model_spec.py",
    )
    report = load_module(
        "report_ols_nonlinearity_smoke",
        skill_root / "generate-final-report" / "scripts" / "generate_report.py",
    )

    x = np.linspace(-3, 3, 90)
    frame = pd.DataFrame({"x": x, "score": 4 + 2.5 * x**2})
    forms = approval.parse_continuous_forms(["x=quadratic"])
    specifications = approval.continuous_form_specifications(
        frame, ["x"], forms, "score", None, "ols"
    )
    item = specifications["x"]
    assert item["form"] == "quadratic"
    assert item["preview_kind"] == "unadjusted-binned-outcome-mean"
    assert len(item["unadjusted_binned_preview"]) >= 4
    assert "unadjusted_binned_rate_preview" not in item
    transformed = approval.apply_continuous_forms(frame[["x"]], specifications)
    assert list(transformed.columns) == ["x", "x__quadratic"]
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()

    summary = {
        "model_type": "ols",
        "continuous_shape_tests": [
            {
                "variable": "x",
                "form": "quadratic",
                "overall_p_value": 0.001,
                "overall_p_value_adjusted_bh": 0.001,
                "nonlinear_p_value": 0.001,
                "nonlinear_p_value_adjusted_bh": 0.001,
            }
        ],
    }
    shape_html = report.build_continuous_shape_html(
        summary,
        {"variable_metadata": {"x": {"display_name": "主要因素"}}},
        0.05,
    )
    assert "调整后预测曲线" in shape_html
    assert "预测概率" not in shape_html
    assert "优势比" not in shape_html

    run_source = (
        skill_root / "run-statistical-analysis" / "scripts" / "run_analysis.py"
    ).read_text(encoding="utf-8")
    for token in (
        'model_type in {"ols", "logistic", "poisson", "negative-binomial"}',
        "adjusted-outcome-",
        "调整后预测结果",
        "continuous_shape_tests",
    ):
        assert token in run_source

    print(
        json.dumps(
            {
                "ok": True,
                "form": item["form"],
                "preview": item["preview_kind"],
                "terms": item["term_names"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
