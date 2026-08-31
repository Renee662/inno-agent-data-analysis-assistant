#!/usr/bin/env python3
"""Verify deterministic automatic functional-form assignment across model families."""

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
        "approve_automatic_forms_smoke",
        skill_root / "run-statistical-analysis" / "scripts" / "approve_model_spec.py",
    )
    rich = pd.DataFrame({"x": np.linspace(0, 20, 80)})
    sparse = pd.DataFrame({"x": np.tile([0, 1, 2, 3], 20)})
    for model_type in (
        "ols", "logistic", "poisson", "negative-binomial",
        "multinomial-logistic", "ordinal-logistic",
    ):
        forms, decisions = approval.resolve_continuous_forms(
            rich, ["x"], {}, {}, model_type
        )
        assert forms == {"x": "restricted-cubic-spline"}
        assert decisions["x"]["source"] == "automatic-fixed-flexible-default"
    forms, decisions = approval.resolve_continuous_forms(
        sparse, ["x"], {}, {}, "ols"
    )
    assert forms == {"x": "linear"}
    assert decisions["x"]["source"] == "automatic-limited-support-linear"
    forms, decisions = approval.resolve_continuous_forms(
        rich,
        ["x"],
        {},
        {
            "x": {
                "functional_form": "quadratic",
                "functional_form_rationale": "理论预先规定检验倒U形关系。",
            }
        },
        "poisson",
    )
    assert forms == {"x": "quadratic"}
    assert decisions["x"]["source"] == "approved-task-metadata"

    multicategory = pd.DataFrame(
        {
            "x": np.linspace(0, 20, 80),
            "y": np.tile(["low", "middle", "high", "middle"], 20),
        }
    )
    category_spec = approval.continuous_form_specifications(
        multicategory,
        ["x"],
        {"x": "restricted-cubic-spline"},
        "y",
        None,
        "ordinal-logistic",
        ["low", "middle", "high"],
    )["x"]
    assert category_spec["preview_kind"] == "unadjusted-binned-category-proportions"
    first_proportions = category_spec["unadjusted_binned_category_preview"][0][
        "observed_category_proportions"
    ]
    assert list(first_proportions) == ["low", "middle", "high"]
    assert abs(sum(first_proportions.values()) - 1.0) < 1e-12

    skill_text = (skill_root / "run-statistical-analysis" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "do not ask them “linear, quadratic, or spline?”" in skill_text
    assert "Never fit several forms and select the smallest p-value" in skill_text
    run_text = (
        skill_root / "run-statistical-analysis" / "scripts" / "run_analysis.py"
    ).read_text(encoding="utf-8")
    assert "adjusted-count-" in run_text
    assert "adjusted-category-probabilities-" in run_text
    assert "调整后预计计数" in run_text
    print(
        json.dumps(
            {
                "ok": True,
                "supported_models": 6,
                "automatic_default": "restricted-cubic-spline",
                "limited_support": "linear-with-limitation",
                "domain_override": "quadratic",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
