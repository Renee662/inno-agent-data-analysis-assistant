#!/usr/bin/env python3
"""Verify the multinomial IIA sensitivity gate and its reporting contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_categories(
    rng: np.random.Generator, logits: np.ndarray, categories: list[str]
) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    uniforms = rng.uniform(size=len(logits))
    cumulative = np.cumsum(probabilities, axis=1)
    codes = (uniforms[:, None] > cumulative).sum(axis=1)
    return np.asarray(categories, dtype=object)[codes]


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    preset = app_root / "presets" / "data-analysis-assistant"
    module = load_module("iia_smoke", preset / ".skills" / "_shared" / "iia.py")
    categories = ["reference", "bus", "rail", "walk"]

    clear_rng = np.random.default_rng(20260822)
    clear_x = clear_rng.normal(size=2400)
    clear_logits = np.column_stack(
        [
            np.zeros(len(clear_x)),
            0.2 + 0.45 * clear_x,
            -0.25 + 0.7 * clear_x,
            -0.1 - 0.5 * clear_x,
        ]
    )
    clear_frame = pd.DataFrame(
        {"x": clear_x, "choice": sample_categories(clear_rng, clear_logits, categories)}
    )
    clear_design = sm.add_constant(clear_frame[["x"]], has_constant="add")
    clear = module.screen_iia(clear_frame, "choice", clear_design, categories, 0.05)
    assert clear["status"] == "clear-no-detected-sensitivity", clear
    assert clear["model_fitting_allowed"] is True
    assert clear["evaluated_deletions"] == 3

    violation_rng = np.random.default_rng(20260823)
    violation_x = violation_rng.uniform(-2.4, 2.4, size=5200)
    violation_logits = np.column_stack(
        [
            np.zeros(len(violation_x)),
            2.8 * (violation_x**2 - 1.2),
            1.25 * violation_x,
            -1.1 * violation_x,
        ]
    )
    violation_frame = pd.DataFrame(
        {
            "x": violation_x,
            "choice": sample_categories(violation_rng, violation_logits, categories),
        }
    )
    violation_design = sm.add_constant(violation_frame[["x"]], has_constant="add")
    violated = module.screen_iia(
        violation_frame, "choice", violation_design, categories, 0.05
    )
    assert violated["status"] == "sensitivity-detected", violated
    assert violated["model_fitting_allowed"] is False
    assert any(
        item.get("decision") == "sensitivity-detected"
        for item in violated["deletion_tests"]
    )

    run_skill = (preset / ".skills" / "run-statistical-analysis" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    report_skill = (preset / ".skills" / "generate-final-report" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "IIA假设是" in run_skill
    assert "never as proof that IIA holds" in report_skill
    print(
        json.dumps(
            {
                "ok": True,
                "clear_minimum_adjusted_p": clear["minimum_adjusted_p_value"],
                "violation_minimum_adjusted_p": violated["minimum_adjusted_p_value"],
                "deletions": violated["total_deletions"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
