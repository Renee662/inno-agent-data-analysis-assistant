"""Canonical model identifiers and display labels for the bundled workflow."""

from __future__ import annotations


SUPPORTED_MODEL_TYPES = (
    "ols",
    "logistic",
    "poisson",
    "negative-binomial",
    "multinomial-logistic",
    "ordinal-logistic",
)

SUPPORTED_MODEL_TYPE_SET = frozenset(SUPPORTED_MODEL_TYPES)

MODEL_LABELS = {
    "ols": "普通最小二乘回归（OLS）",
    "logistic": "二元 Logistic 回归",
    "poisson": "Poisson 回归",
    "negative-binomial": "负二项回归",
    "multinomial-logistic": "多分类 Logistic 回归",
    "ordinal-logistic": "有序 Logistic 回归",
}
