#!/usr/bin/env python3
"""Smoke-test excess-zero and negative-binomial necessity gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SHARED = (
    ROOT
    / "apps/inno-agent/presets/data-analysis-assistant/.skills/_shared"
)
sys.path.insert(0, str(SHARED))

from negative_binomial_need import screen_negative_binomial_need  # noqa: E402
from zero_inflation import screen_zero_inflation  # noqa: E402


def frame_and_design(outcome: np.ndarray, x: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame({"count": outcome.astype(int), "x": x})
    design = pd.DataFrame({"const": np.ones(len(x)), "x": x})
    return frame, design


def main() -> int:
    rng = np.random.default_rng(20260822)
    n = 1800
    x = rng.normal(size=n)
    mu = np.exp(0.35 + 0.25 * x)

    poisson_y = rng.poisson(mu)
    poisson_frame, design = frame_and_design(poisson_y, x)
    poisson_zero = screen_zero_inflation(
        poisson_frame, "count", design, "poisson", 0.05
    )
    assert poisson_zero["status"] == "clear-no-detected-excess-zeros"
    nb_not_needed = screen_negative_binomial_need(
        poisson_frame, "count", design, 0.05
    )
    assert nb_not_needed["status"] == "no-detected-need-for-extra-dispersion"
    assert nb_not_needed["model_fitting_allowed"] is False

    structural_zero = rng.random(n) < 0.45
    zero_inflated_y = rng.poisson(mu)
    zero_inflated_y[structural_zero] = 0
    zero_frame, zero_design = frame_and_design(zero_inflated_y, x)
    excess_zero = screen_zero_inflation(
        zero_frame, "count", zero_design, "poisson", 0.05
    )
    assert excess_zero["status"] == "excess-zeros-detected"
    assert excess_zero["model_fitting_allowed"] is False

    alpha = 1.1
    shape = 1.0 / alpha
    scale = alpha * mu
    latent_rate = rng.gamma(shape=shape, scale=scale)
    nb_y = rng.poisson(latent_rate)
    nb_frame, nb_design = frame_and_design(nb_y, x)
    nb_needed = screen_negative_binomial_need(nb_frame, "count", nb_design, 0.05)
    assert nb_needed["status"] == "extra-dispersion-supported"
    assert nb_needed["model_fitting_allowed"] is True

    print(
        json.dumps(
            {
                "ok": True,
                "ordinary_poisson_zero_status": poisson_zero["status"],
                "negative_binomial_on_poisson_status": nb_not_needed["status"],
                "zero_inflated_status": excess_zero["status"],
                "negative_binomial_needed_status": nb_needed["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
