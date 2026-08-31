#!/usr/bin/env python3
"""Deterministic smoke checks for the ordinal equal-slopes gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "presets" / "data-analysis-assistant" / ".skills" / "_shared"
sys.path.insert(0, str(SHARED))

from proportional_odds import materially_matches, screen_proportional_odds  # noqa: E402


def sample_categories(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    uniforms = rng.uniform(size=len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    return (uniforms[:, None] > cumulative).sum(axis=1)


def main() -> None:
    rng = np.random.default_rng(20260817)
    n = 2400
    x = rng.normal(size=n)
    design = pd.DataFrame({"x": x})

    first = 1 / (1 + np.exp(-(-0.8 - 0.9 * x)))
    second = 1 / (1 + np.exp(-(0.9 - 0.9 * x)))
    clear_codes = sample_categories(
        rng, np.column_stack([first, second - first, 1 - second])
    )
    clear_frame = pd.DataFrame(
        {"y": np.asarray(["low", "middle", "high"])[clear_codes]}
    )
    clear = screen_proportional_odds(
        clear_frame, "y", design, ["low", "middle", "high"], 0.05
    )
    if clear["status"] != "clear-no-detected-violation" or clear["model_fitting_allowed"] is not True:
        raise AssertionError(f"True proportional-odds data was blocked: {clear}")
    if not materially_matches(clear, dict(clear)):
        raise AssertionError("Identical proportional-odds evidence did not match")

    first_bad = 1 / (1 + np.exp(-(-0.8 - 2.0 * x)))
    second_bad = 1 / (1 + np.exp(-(0.9 + 1.2 * x)))
    # Enforce ordered cumulative probabilities while retaining strong slope drift.
    lower = np.minimum(first_bad, second_bad - 0.02)
    lower = np.clip(lower, 0.001, 0.979)
    upper = np.maximum(second_bad, lower + 0.02)
    upper = np.clip(upper, 0.021, 0.999)
    bad_codes = sample_categories(
        rng, np.column_stack([lower, upper - lower, 1 - upper])
    )
    bad_frame = pd.DataFrame(
        {"y": np.asarray(["low", "middle", "high"])[bad_codes]}
    )
    violated = screen_proportional_odds(
        bad_frame, "y", design, ["low", "middle", "high"], 0.05
    )
    if violated["status"] != "violation-detected" or violated["model_fitting_allowed"] is not False:
        raise AssertionError(f"Nonparallel slopes were not blocked: {violated}")
    drifted = dict(clear)
    drifted["p_value"] = float(clear["p_value"]) / 2
    if materially_matches(clear, drifted):
        raise AssertionError("Changed runtime evidence was not detected")

    print(
        {
            "status": "ok",
            "clear_p_value": clear["p_value"],
            "violation_p_value": violated["p_value"],
        }
    )


if __name__ == "__main__":
    main()
