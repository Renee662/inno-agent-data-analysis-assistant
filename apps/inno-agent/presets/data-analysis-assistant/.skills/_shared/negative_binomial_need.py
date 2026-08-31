"""Adjusted gate for whether a negative-binomial model needs extra dispersion."""

from __future__ import annotations

from typing import Any

from count_dispersion import materially_matches as dispersion_matches
from count_dispersion import screen_count_dispersion


def not_applicable() -> dict[str, Any]:
    return {
        "status": "not-applicable",
        "model_fitting_allowed": True,
        "plain_explanation": "负二项必要性检查只适用于拟采用负二项回归的计数结果。",
    }


def screen_negative_binomial_need(*args: Any, **kwargs: Any) -> dict[str, Any]:
    dispersion = screen_count_dispersion(*args, **kwargs)
    supported = dispersion["status"] == "overdispersion-detected"
    return {
        **dispersion,
        "status": "extra-dispersion-supported" if supported else "no-detected-need-for-extra-dispersion",
        "model_fitting_allowed": supported,
        "test": "one-sided adjusted NB2 extra-dispersion necessity screen",
        "decision": "negative-binomial-supported" if supported else "poisson-variance-not-rejected",
        "plain_explanation": (
            "该检查先按已批准因素评估计数波动是否显著超过Poisson允许的范围。"
            "只有发现调整后的额外离散证据时，负二项增加的离散参数才有当前数据支持。"
        ),
        "interpretation_boundary": (
            "未发现额外离散不证明Poisson绝对正确；但当前数据不足以支持直接增加负二项离散参数。"
            "系统不会据此自动换模，必须返回模型选择并重新批准。"
        ),
        "required_action": (
            "negative-binomial modeling may proceed with the recorded evidence"
            if supported
            else "return to model choice for an explicitly approved Poisson model or stop"
        ),
    }


def materially_matches(approved: dict[str, Any], runtime: dict[str, Any]) -> bool:
    restored_approved = {
        **approved,
        "status": (
            "overdispersion-detected"
            if approved.get("status") == "extra-dispersion-supported"
            else "clear-no-detected-overdispersion"
        ),
        "model_fitting_allowed": approved.get("status") != "extra-dispersion-supported",
        "test": "one-sided Cameron-Trivedi-style NB2 auxiliary score screen",
        "decision": (
            "positive-overdispersion-detected"
            if approved.get("status") == "extra-dispersion-supported"
            else "do-not-reject-poisson-variance"
        ),
    }
    restored_runtime = {
        **runtime,
        "status": (
            "overdispersion-detected"
            if runtime.get("status") == "extra-dispersion-supported"
            else "clear-no-detected-overdispersion"
        ),
        "model_fitting_allowed": runtime.get("status") != "extra-dispersion-supported",
        "test": "one-sided Cameron-Trivedi-style NB2 auxiliary score screen",
        "decision": (
            "positive-overdispersion-detected"
            if runtime.get("status") == "extra-dispersion-supported"
            else "do-not-reject-poisson-variance"
        ),
    }
    return dispersion_matches(restored_approved, restored_runtime)
