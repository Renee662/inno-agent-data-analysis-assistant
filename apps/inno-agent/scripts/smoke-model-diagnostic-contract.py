#!/usr/bin/env python3
"""Verify model-specific diagnostic reporting contracts without fitting models."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_report_diagnostic_smoke", path)
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
    task = {"variable_metadata": {"y": {"display_name": "结果"}}}
    cases = {
        "ols": {
            "metrics": {
                "r_squared": 0.4,
                "rmse": 1.2,
                "breusch_pagan_p_value": 0.3,
                "jarque_bera_p_value": 0.2,
            },
            "applicability": {"normal_qq": "applicable", "heteroskedasticity": "applicable"},
            "required": ["Breusch", "Jarque"],
        },
        "logistic": {
            "metrics": {"roc_auc": 0.78, "brier_score": 0.18, "calibration_mean_absolute_error": 0.04},
            "applicability": {"normal_qq": "not-applicable", "heteroskedasticity": "not-applicable"},
            "required": ["ROC AUC", "Brier", "不适用"],
        },
        "poisson": {
            "metrics": {"dispersion": 1.1, "count_rmse": 2.0},
            "applicability": {"normal_qq": "not-applicable", "heteroskedasticity": "not-applicable"},
            "required": ["离散度", "不适用"],
        },
        "negative-binomial": {
            "metrics": {"negative_binomial_alpha": 0.6, "count_rmse": 2.0},
            "applicability": {"normal_qq": "not-applicable", "heteroskedasticity": "not-applicable"},
            "required": ["负二项", "不适用"],
        },
        "multinomial-logistic": {
            "metrics": {"classification_accuracy": 0.7, "multiclass_log_loss": 0.8},
            "applicability": {"normal_qq": "not-applicable", "heteroskedasticity": "not-applicable"},
            "required": ["多分类", "混淆矩阵", "不适用"],
        },
        "ordinal-logistic": {
            "metrics": {"classification_accuracy": 0.7, "ordinal_mean_absolute_category_error": 0.4},
            "applicability": {"normal_qq": "not-applicable", "heteroskedasticity": "not-applicable"},
            "required": ["有序", "比例优势", "不适用"],
        },
    }
    for model_type, case in cases.items():
        summary = {
            "model_type": model_type,
            "metrics": case["metrics"],
            "diagnostic_applicability": case["applicability"],
            "influence_available": False,
            "influential_cleaned_data_rows": [],
        }
        html = report.build_diagnostics_html(
            summary, [], {"model_type": model_type, "outcome": "y"}, task
        )
        for required in case["required"]:
            assert required in html, (model_type, required)
        assert "未能评估" in html
        assert "不能把“未计算”解释为“没有高影响记录”" in html
        if model_type != "ols":
            assert "残差分布异常" not in html
    required_figures = {
        "logistic-roc.png",
        "logistic-calibration.png",
        "count-residuals-vs-fitted.png",
        "count-observed-vs-fitted.png",
        "classification-confusion.png",
        "classification-calibration.png",
    }
    assert required_figures.issubset(report.FIGURE_METADATA)
    print(json.dumps({"ok": True, "models": list(cases), "figures": sorted(required_figures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
