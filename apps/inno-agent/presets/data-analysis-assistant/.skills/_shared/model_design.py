"""Shared design-matrix transformations for model approval and execution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_continuous_forms(
    frame: pd.DataFrame, specifications: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Apply approved polynomial or restricted-cubic-spline terms."""
    transformed = frame.copy()
    for column, item in specifications.items():
        values = pd.to_numeric(transformed[column], errors="coerce")
        standardized = (values - float(item["center"])) / float(item["scale"])
        form = str(item["form"])
        if form == "quadratic":
            transformed[f"{column}__quadratic"] = standardized**2
        elif form == "restricted-cubic-spline":
            knots = [float(value) for value in item["knots_standardized"]]
            penultimate, last = knots[-2], knots[-1]
            positive_cube = lambda value, knot: np.maximum(value - knot, 0.0) ** 3
            for basis_index, knot in enumerate(knots[:-2], start=1):
                basis = (
                    (positive_cube(standardized, knot) - positive_cube(standardized, penultimate))
                    / (last - knot)
                    - (positive_cube(standardized, penultimate) - positive_cube(standardized, last))
                    / (last - penultimate)
                )
                transformed[f"{column}__rcs{basis_index}"] = basis
    return transformed
