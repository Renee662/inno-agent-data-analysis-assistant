#!/usr/bin/env python3
"""Smoke-test EDA plots for numeric-coded categorical variables."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_report_eda_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    report_script = (
        app_root
        / "presets"
        / "data-analysis-assistant"
        / ".skills"
        / "generate-final-report"
        / "scripts"
        / "generate_report.py"
    )
    report = load_module(report_script)
    with tempfile.TemporaryDirectory(prefix="inno-eda-roles-") as temp_value:
        temp = Path(temp_value)
        data_path = temp / "cleaned-data.csv"
        figure_dir = temp / "figures"
        figure_dir.mkdir()
        rows = 120
        pd.DataFrame(
            {
                "income_code": [1 if index % 4 == 0 else 0 for index in range(rows)],
                "class_code": [1 + index % 3 for index in range(rows)],
                "level_code": [1 + index % 3 for index in range(rows)],
                "age": [20 + index % 50 for index in range(rows)],
                "workclass_code": [1 + index % 3 for index in range(rows)],
            }
        ).to_csv(data_path, index=False, encoding="utf-8-sig")
        task = {
            "variables": [
                {"column": "income_code", "inferred_type": "numeric"},
                {"column": "age", "inferred_type": "categorical"},
                {"column": "workclass_code", "inferred_type": "numeric"},
            ],
            "variable_metadata": {
                "income_code": {"display_name": "年收入是否超过5万美元"},
                "age": {"display_name": "年龄", "unit": "岁"},
                "workclass_code": {
                    "display_name": "工作类型",
                    "category_meanings": {"1": "私营", "2": "政府", "3": "其他"},
                },
            },
        }
        model_spec = {
            "outcome": "income_code",
            "predictors": ["age", "workclass_code"],
            "controls": [],
            "model_type": "logistic",
            "positive_class": 1,
            "categorical_columns": ["workclass_code"],
        }
        html, generated = report.build_eda_artifacts(
            data_path, task, model_spec, figure_dir, False
        )
        assert "二分类" in html
        assert "连续数值" in html
        assert "无序多分类" in html
        assert "中位数" in html and "众数" in html and "四分位数" in html
        assert "众数为" in html
        assert report.eda_variable_roles(
            task, model_spec, ["income_code", "age", "workclass_code"]
        )["age"] == "continuous"
        assert generated == ["eda-distributions.png", "eda-relationships.png"], generated
        assert all((figure_dir / name).stat().st_size > 0 for name in generated)

        classification_roles = {"logistic": "binary"}
        for model_type, outcome, expected_role, role_label, extra in (
            (
                "multinomial-logistic",
                "class_code",
                "nominal",
                "无序多分类",
                {"reference_class": 1},
            ),
            (
                "ordinal-logistic",
                "level_code",
                "ordinal",
                "有序多分类",
                {"outcome_categories": [1, 2, 3]},
            ),
        ):
            current_task = {
                "variables": [
                    {"column": outcome, "inferred_type": "numeric"},
                    {"column": "age", "inferred_type": "numeric"},
                    {"column": "workclass_code", "inferred_type": "numeric"},
                ],
                "variable_metadata": {
                    outcome: {"display_name": "测试类别"},
                    **task["variable_metadata"],
                },
            }
            current_spec = {
                "outcome": outcome,
                "predictors": ["age", "workclass_code"],
                "controls": [],
                "model_type": model_type,
                "categorical_columns": ["workclass_code"],
                **extra,
            }
            current_dir = temp / f"figures-{model_type}"
            current_dir.mkdir()
            current_html, current_generated = report.build_eda_artifacts(
                data_path, current_task, current_spec, current_dir, False
            )
            current_roles = report.eda_variable_roles(
                current_task, current_spec, [outcome, "age", "workclass_code"]
            )
            assert current_roles[outcome] == expected_role
            assert role_label in current_html
            assert current_generated == [
                "eda-distributions.png",
                "eda-relationships.png",
            ]
            assert all((current_dir / name).stat().st_size > 0 for name in current_generated)
            classification_roles[model_type] = current_roles[outcome]
        print(
            json.dumps(
                {
                    "ok": True,
                    "generated": generated,
                    "roles": report.eda_variable_roles(
                        task,
                        model_spec,
                        ["income_code", "age", "workclass_code"],
                    ),
                    "classification_outcome_roles": classification_roles,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
