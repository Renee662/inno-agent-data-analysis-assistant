#!/usr/bin/env python3
"""Deterministic checks for the ordinary-Poisson overdispersion gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "presets" / "data-analysis-assistant" / ".skills" / "_shared"
sys.path.insert(0, str(SHARED))

from count_dispersion import materially_matches, screen_count_dispersion  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20260817)
    n = 2400
    x = rng.normal(size=n)
    design = sm.add_constant(pd.DataFrame({"x": x}), has_constant="add")
    mean = np.exp(0.3 + 0.45 * x)

    poisson_frame = pd.DataFrame({"count": rng.poisson(mean)})
    clear = screen_count_dispersion(poisson_frame, "count", design, 0.05)
    if clear["status"] != "clear-no-detected-overdispersion" or clear["model_fitting_allowed"] is not True:
        raise AssertionError(f"True Poisson data was blocked: {clear}")
    if not materially_matches(clear, dict(clear)):
        raise AssertionError("Identical count-dispersion evidence did not match")

    size = 0.65
    probability = size / (size + mean)
    overdispersed_frame = pd.DataFrame(
        {"count": rng.negative_binomial(size, probability)}
    )
    blocked = screen_count_dispersion(overdispersed_frame, "count", design, 0.05)
    if blocked["status"] != "overdispersion-detected" or blocked["model_fitting_allowed"] is not False:
        raise AssertionError(f"Overdispersed counts were not blocked: {blocked}")

    print(
        {
            "status": "ok",
            "clear_p_value": clear["p_value"],
            "blocked_p_value": blocked["p_value"],
            "blocked_pearson_dispersion": blocked["pearson_dispersion_descriptive"],
        }
    )


if __name__ == "__main__":
    main()
