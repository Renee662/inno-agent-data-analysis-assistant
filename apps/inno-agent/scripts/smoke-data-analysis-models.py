#!/usr/bin/env python3
"""Offline smoke test for every bundled data-analysis regression family."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "presets" / "data-analysis-assistant"
APPROVE = (
    PRESET
    / ".skills"
    / "run-statistical-analysis"
    / "scripts"
    / "approve_model_spec.py"
)
FINALIZE = (
    PRESET
    / ".skills"
    / "run-statistical-analysis"
    / "scripts"
    / "finalize_model_spec.py"
)
ASSESS = (
    PRESET
    / ".skills"
    / "run-statistical-analysis"
    / "scripts"
    / "assess_workflow_support.py"
)
RUN = (
    PRESET
    / ".skills"
    / "run-statistical-analysis"
    / "scripts"
    / "run_analysis.py"
)
REPORT = (
    PRESET
    / ".skills"
    / "generate-final-report"
    / "scripts"
    / "generate_report.py"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_approval(path: Path, action: str, artifact: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "approvalId": f"smoke-{action}",
                "status": "approved",
                "action": action,
                "source": "web-question-dialog",
                "questionId": f"question-{action}",
                "sessionId": "smoke-session",
                "approvedAt": "2026-08-16T00:00:00+08:00",
                "artifact": {"path": artifact.name, "sha256": file_sha256(artifact)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def no_missingness_contract() -> dict[str, object]:
    return {
        "missingness_bias_screen": {
            "status": "clear-no-selected-missingness",
            "diagnostic_scope": "descriptive-sample-composition",
            "complete_case_rows": 0,
            "complete_case_excluded_rows": 0,
            "complete_case_excluded_rate": 0.0,
            "complete_case_outcome_retained": {"kind": "categorical", "distribution": []},
            "complete_case_outcome_excluded": {"kind": "categorical", "distribution": []},
            "field_comparisons": [],
            "interpretation": {
                "can_identify_mcar_mar_mnar": False,
                "can_prove_no_selection_bias": False,
                "model_estimate_sensitivity_included": False,
            },
        },
        "missingness_bias_rows": [],
        "post_preparation_missingness": {
            "rows": 0,
            "rows_with_any_selected_missingness": 0,
            "by_selected_column": {},
        },
        "missingness_conclusion_contract": {
            "status": "clear",
            "scope": "full-selected-sample",
            "model_estimate_sensitivity_completed": False,
        },
    }


def invoke(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def invoke_expect_failure(*args: str, expected: str) -> None:
    completed = subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode == 0:
        raise AssertionError(f"Expected command failure: {' '.join(args)}")
    if expected.lower() not in combined.lower():
        raise AssertionError(
            f"Failure did not mention {expected!r}: {' '.join(args)}\n{combined}"
        )


def main() -> None:
    rng = np.random.default_rng(20260725)
    # Spline-enabled multicategory models estimate more terms per outcome equation;
    # keep the synthetic support comfortably above the workflow's review benchmark.
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.choice(["A", "B"], size=n)
    linear = 0.35 + 0.75 * x1 + 0.4 * (x2 == "B")
    logistic_probability = 1 / (1 + np.exp(-linear))
    multinomial_score = linear + rng.normal(scale=0.8, size=n)
    ordinal_score = linear + rng.logistic(scale=0.7, size=n)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y_ols": linear + rng.normal(scale=0.7, size=n),
            "y_logistic": np.where(
                rng.uniform(size=n) < logistic_probability, "yes", "no"
            ),
            "y_poisson": np.random.default_rng(1).poisson(
                np.exp(np.clip(linear, -1.5, 1.5))
            ),
            "y_negative_binomial": rng.negative_binomial(
                2.0, 2.0 / (2.0 + np.exp(np.clip(linear, -1.5, 1.5)))
            ),
            "y_multinomial": np.where(
                multinomial_score < -0.25,
                "low",
                np.where(multinomial_score < 0.85, "middle", "high"),
            ),
            "y_ordinal": np.where(
                ordinal_score < -0.35,
                "low",
                np.where(ordinal_score < 0.9, "middle", "high"),
            ),
        }
    )
    cases = [
        ("ols", "y_ols", []),
        ("logistic", "y_logistic", ["--positive-class", "yes"]),
        ("poisson", "y_poisson", []),
        ("negative-binomial", "y_negative_binomial", []),
        (
            "multinomial-logistic",
            "y_multinomial",
            ["--reference-class", "low"],
        ),
        (
            "ordinal-logistic",
            "y_ordinal",
            ["--category-order", "low", "middle", "high"],
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="inno-data-analysis-smoke-") as temp:
        root = Path(temp)
        conversation_root = root / "conversations" / "2026-07-26_smoke"
        work_root = conversation_root / "work"
        outputs_root = conversation_root / "outputs"
        work_root.mkdir(parents=True)
        os.environ["INNO_CONVERSATION_DIR"] = str(conversation_root)
        data_path = root / "cleaned-data.csv"
        prep_path = work_root / "data-preparation-log.json"
        missingness_path = work_root / "missingness-impact.csv"
        profile_path = work_root / "data-profile.json"
        frame.to_csv(data_path, index=False, encoding="utf-8-sig")
        prep_path.write_text(
            json.dumps(
                {
                    "data_preparation_executed": True,
                    "modeling_executed": False,
                    **no_missingness_contract(),
                    "analysis_metadata": {
                        "analysis_exclusions": [],
                        "categorical_columns": ["x2"],
                    },
                }
            ),
            encoding="utf-8",
        )
        missingness_path.write_text(
            "column,role,missing_rows,observed_rows,max_outcome_proportion_gap,outcome_mean_difference\n",
            encoding="utf-8-sig",
        )
        profile_path.write_text(
            json.dumps(
                {
                    "source_file": "cleaned-data.csv",
                    "profiles": [
                        {
                            "sheet": "data",
                            "row_count": n,
                            "column_count": len(frame.columns),
                            "columns": [
                                {
                                    "column": column,
                                    "inferred_type": (
                                        "numeric"
                                        if pd.api.types.is_numeric_dtype(frame[column])
                                        else "categorical"
                                    ),
                                    "missing_rate": 0,
                                    "sensitive_name_candidate": False,
                                }
                                for column in frame.columns
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        completed_models: list[str] = []
        for model_type, outcome, extra in cases:
            task_path = work_root / f"task-{model_type}.json"
            approval_dir = work_root / f"approval-{model_type}"
            result_dir = work_root / f"results-{model_type}"
            task_path.write_text(
                json.dumps(
                    {
                        "status": "approved",
                        "approval": {"approval_id": "smoke-task"},
                        "title": model_type,
                        "goal": "prediction" if model_type == "logistic" else "association",
                        "report_title": f"{model_type} 冒烟测试报告",
                        "dataset_summary": "这是一份用于离线验证统计分析流程的合成数据。",
                        "research_question": "主要因素与测试结果之间有什么关系？",
                        "outcome": outcome,
                        "predictors": ["x1"],
                        "controls": ["x2"],
                        "variables": [
                            {"column": outcome, "inferred_type": "numeric"},
                            {"column": "x1", "inferred_type": "numeric"},
                            {"column": "x2", "inferred_type": "categorical"},
                        ],
                        "variable_metadata": {
                            outcome: {
                                "display_name": "测试结果",
                                "unit": "单位",
                                "interpretation_increment": 1,
                            },
                            "x1": {
                                "display_name": "主要因素",
                                "unit": "单位",
                                "interpretation_increment": 1,
                            },
                            "x2": {
                                "display_name": "分组因素",
                                "unit": "类别",
                                "reference_category": "A",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            workflow_input = approval_dir / "workflow-decision-input.json"
            workflow_input.parent.mkdir(parents=True, exist_ok=True)
            workflow_input.write_text(
                json.dumps(
                    {
                        "observation_structure": "independent",
                        "outcome_process": "standard",
                        "sampling_design": "simple",
                        "count_exposure": "not-applicable",
                    }
                ),
                encoding="utf-8",
            )
            workflow_approval = write_approval(
                approval_dir / "workflow-approval.json",
                "approve-workflow-support",
                workflow_input,
            )
            invoke(
                str(ASSESS),
                "--task",
                str(task_path),
                "--decision-input",
                str(workflow_input),
                "--approval-record",
                str(workflow_approval),
                "--output-dir",
                str(approval_dir),
            )
            invoke(
                str(APPROVE),
                "--task",
                str(task_path),
                "--workflow-support",
                str(approval_dir / "workflow-support-assessment.json"),
                "--preparation-log",
                str(prep_path),
                "--data",
                str(data_path),
                "--model-type",
                model_type,
                "--output-dir",
                str(approval_dir),
                *extra,
            )
            proposal = approval_dir / "model-specification-proposal.json"
            proposal_payload = json.loads(proposal.read_text(encoding="utf-8"))
            category_support = proposal_payload.get("category_support_screen", {})
            if model_type in {"logistic", "multinomial-logistic", "ordinal-logistic"}:
                if category_support.get("status") != "clear" or category_support.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(
                        f"Unexpected category-support block for {model_type}: {category_support}"
                    )
            if proposal_payload.get("categorical_reference_categories") != {"x2": "A"}:
                raise AssertionError(
                    f"Approved predictor reference was not proposed for {model_type}: "
                    f"{proposal_payload.get('categorical_reference_categories')}"
                )
            proposal_iia = proposal_payload.get("iia_check", {})
            if model_type == "multinomial-logistic":
                if proposal_iia.get("status") != "clear-no-detected-sensitivity" or proposal_iia.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(f"Multinomial IIA proposal gate did not pass: {proposal_iia}")
            elif proposal_iia.get("status") != "not-applicable":
                raise AssertionError(f"Non-multinomial IIA proposal record is invalid: {model_type}")
            proposal_nb_need = proposal_payload.get("negative_binomial_need_check", {})
            if model_type == "negative-binomial":
                if proposal_nb_need.get("status") != "extra-dispersion-supported" or proposal_nb_need.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(f"Negative-binomial need proposal gate did not pass: {proposal_nb_need}")
            elif proposal_nb_need.get("status") != "not-applicable":
                raise AssertionError(f"Unexpected negative-binomial need proposal record: {model_type}")
            proposal_zero = proposal_payload.get("zero_inflation_check", {})
            if model_type in {"poisson", "negative-binomial"}:
                if proposal_zero.get("status") != "clear-no-detected-excess-zeros" or proposal_zero.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(f"Excess-zero proposal gate did not pass for {model_type}: {proposal_zero}")
            elif proposal_zero.get("status") != "not-applicable":
                raise AssertionError(f"Unexpected excess-zero proposal record: {model_type}")
            if model_type == "logistic":
                if proposal_payload.get("positive_class") != "yes" or proposal_payload.get(
                    "outcome_reference_class"
                ) != "no":
                    raise AssertionError("Logistic target/reference classes are incomplete")
            model_approval = write_approval(
                approval_dir / "model-approval.json",
                "approve-model-specification",
                proposal,
            )
            invoke(
                str(FINALIZE),
                "--proposal",
                str(proposal),
                "--approval-record",
                str(model_approval),
                "--task",
                str(task_path),
                "--workflow-support",
                str(approval_dir / "workflow-support-assessment.json"),
                "--preparation-log",
                str(prep_path),
                "--data",
                str(data_path),
                "--output-dir",
                str(approval_dir),
            )
            result = invoke(
                str(RUN),
                "--data",
                str(data_path),
                "--spec",
                str(approval_dir / "approved-model-specification.json"),
                "--preparation-log",
                str(prep_path),
                "--output-dir",
                str(result_dir),
            )
            if result.get("model_type") != model_type:
                raise AssertionError(f"Unexpected model result: {result}")
            model_summary = json.loads(
                (result_dir / "model-summary.json").read_text(encoding="utf-8")
            )
            if model_summary.get("categorical_reference_categories") != {"x2": "A"}:
                raise AssertionError(
                    f"Runtime predictor reference drifted for {model_type}"
                )
            multiplicity = model_summary.get("multiplicity", {})
            if multiplicity.get("coefficient_method") != "Benjamini-Hochberg FDR":
                raise AssertionError(f"Multiplicity correction missing for {model_type}")
            collinearity = model_summary.get("collinearity", {})
            if collinearity.get("status") not in {"clear", "review", "severe"}:
                raise AssertionError(f"Collinearity decision missing for {model_type}")
            influence = model_summary.get("influence_diagnostics", {})
            if influence.get("status") != "available":
                raise AssertionError(f"Influence diagnostics unavailable for {model_type}: {influence}")
            if influence.get("automatic_deletion_performed") is not False:
                raise AssertionError(f"Influence screen deleted records for {model_type}")
            if not (result_dir / "influence-diagnostics.csv").is_file():
                raise AssertionError(f"Influence audit table missing for {model_type}")
            if model_type in {"multinomial-logistic", "ordinal-logistic"} and "不声称具有Cook距离" not in str(influence.get("method")):
                raise AssertionError(f"Classification influence method overclaims Cook distance: {model_type}")
            proportional_odds = model_summary.get("proportional_odds_check", {})
            if model_type == "ordinal-logistic":
                if proportional_odds.get("status") != "clear-no-detected-violation" or proportional_odds.get("model_fitting_allowed") is not True:
                    raise AssertionError(
                        f"Ordinal proportional-odds gate did not pass: {proportional_odds}"
                    )
            elif proportional_odds.get("status") != "not-applicable":
                raise AssertionError(
                    f"Non-ordinal proportional-odds record is invalid: {model_type}"
                )
            iia = model_summary.get("iia_check", {})
            if model_type == "multinomial-logistic":
                if iia.get("status") != "clear-no-detected-sensitivity" or iia.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(f"Multinomial IIA runtime gate did not pass: {iia}")
            elif iia.get("status") != "not-applicable":
                raise AssertionError(f"Non-multinomial IIA runtime record is invalid: {model_type}")
            if not (result_dir / "iia-check.json").is_file():
                raise AssertionError(f"IIA audit record missing for {model_type}")
            count_dispersion = model_summary.get("count_dispersion_check", {})
            if model_type == "poisson":
                if count_dispersion.get("status") != "clear-no-detected-overdispersion" or count_dispersion.get("model_fitting_allowed") is not True:
                    raise AssertionError(
                        f"Poisson overdispersion gate did not pass: {count_dispersion}"
                    )
            elif count_dispersion.get("status") != "not-applicable":
                raise AssertionError(
                    f"Non-Poisson count-dispersion record is invalid: {model_type}"
                )
            if not (result_dir / "count-dispersion-check.json").is_file():
                raise AssertionError(f"Count-dispersion audit record missing for {model_type}")
            nb_need = model_summary.get("negative_binomial_need_check", {})
            if model_type == "negative-binomial":
                if nb_need.get("status") != "extra-dispersion-supported" or nb_need.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(f"Negative-binomial runtime need gate did not pass: {nb_need}")
            elif nb_need.get("status") != "not-applicable":
                raise AssertionError(f"Unexpected negative-binomial runtime need record: {model_type}")
            zero_check = model_summary.get("zero_inflation_check", {})
            if model_type in {"poisson", "negative-binomial"}:
                if zero_check.get("status") != "clear-no-detected-excess-zeros" or zero_check.get(
                    "model_fitting_allowed"
                ) is not True:
                    raise AssertionError(f"Excess-zero runtime gate did not pass for {model_type}: {zero_check}")
            elif zero_check.get("status") != "not-applicable":
                raise AssertionError(f"Unexpected excess-zero runtime record: {model_type}")
            predictive = model_summary.get("predictive_validation", {})
            if model_type == "logistic":
                if predictive.get("status") != "completed" or predictive.get(
                    "model_performance_claim_allowed"
                ) is not True:
                    raise AssertionError(f"Prediction cross-validation did not complete: {predictive}")
            elif predictive.get("status") != "not-applicable":
                raise AssertionError(f"Non-prediction model has invalid validation record: {model_type}")
            for artifact in (
                "negative-binomial-need-check.json",
                "zero-inflation-check.json",
                "predictive-validation.json",
            ):
                if not (result_dir / artifact).is_file():
                    raise AssertionError(f"Missing {artifact} for {model_type}")
            factor_tests = pd.read_csv(result_dir / "factor-omnibus-tests.csv")
            if factor_tests["factor"].tolist() != ["x2"]:
                raise AssertionError(f"Categorical omnibus test missing for {model_type}")
            model_results = pd.read_csv(result_dir / "model-results.csv")
            required_multiplicity_columns = {
                "p_value_adjusted_bh",
                "factor_omnibus_p_value_adjusted_bh",
                "multiplicity_supported",
                "vif",
                "collinearity_restricted",
                "interpretation_supported",
            }
            if not required_multiplicity_columns.issubset(model_results.columns):
                raise AssertionError(
                    f"Multiplicity result columns missing for {model_type}"
                )
            expected_diagnostics = {
                "ols": {"residuals-vs-fitted.png", "residual-qq.png", "adjusted-outcome-x1.png"},
                "logistic": {"logistic-roc.png", "logistic-calibration.png", "adjusted-probability-x1.png"},
                "poisson": {"count-residuals-vs-fitted.png", "count-observed-vs-fitted.png", "adjusted-count-x1.png"},
                "negative-binomial": {"count-residuals-vs-fitted.png", "count-observed-vs-fitted.png", "adjusted-count-x1.png"},
                "multinomial-logistic": {"classification-confusion.png", "classification-calibration.png", "adjusted-category-probabilities-x1.png"},
                "ordinal-logistic": {"classification-confusion.png", "classification-calibration.png", "adjusted-category-probabilities-x1.png"},
            }[model_type]
            actual_diagnostics = set(model_summary.get("diagnostic_figures", []))
            if not expected_diagnostics.issubset(actual_diagnostics):
                raise AssertionError(
                    f"Wrong diagnostic figures for {model_type}: {actual_diagnostics}"
                )
            if model_type != "ols" and {
                "residuals-vs-fitted.png",
                "residual-qq.png",
            } & actual_diagnostics:
                raise AssertionError(f"Linear-model diagnostics leaked into {model_type}")
            applicability = model_summary.get("diagnostic_applicability", {})
            if model_type != "ols" and applicability.get("normal_qq") != "not-applicable":
                raise AssertionError(f"Normal Q-Q applicability wrong for {model_type}")
            prompts = json.loads(
                (result_dir / "learning-prompts.json").read_text(encoding="utf-8")
            )
            prompt_sequence = prompts.get("sequence", [])
            if prompt_sequence[-2:] != ["observation", "diagnostic-reasoning"]:
                raise AssertionError(f"Missing learning prompts for {model_type}")
            if influence.get("candidate_count", 0) and "influence-review" not in prompt_sequence:
                raise AssertionError(f"Influence review prompt missing for {model_type}")
            if model_type == "multinomial-logistic":
                observation = prompts.get("observation", {})
                observation_text = json.dumps(observation, ensure_ascii=False)
                if "类别比例" not in observation_text:
                    raise AssertionError("Multinomial prompt must compare category proportions")
                decision_text = json.dumps(
                    {
                        "question": observation.get("question"),
                        "options": observation.get("options"),
                        "evidence_answer": observation.get("evidence_answer"),
                    },
                    ensure_ascii=False,
                )
                for forbidden in ("Spearman", "正相关", "负相关", "类别顺序编码"):
                    if forbidden in decision_text:
                        raise AssertionError(
                            f"Multinomial prompt still imposes category order: {forbidden}"
                        )
            if model_type in {
                "ols",
                "poisson",
                "negative-binomial",
                "multinomial-logistic",
                "ordinal-logistic",
            }:
                report_dir = outputs_root / model_type
                report_approval = write_approval(
                    approval_dir / "report-approval.json",
                    "approve-final-report",
                    result_dir / "model-summary.json",
                )
                invoke(
                    str(REPORT),
                    "--profile",
                    str(profile_path),
                    "--analysis-task",
                    str(task_path),
                    "--preparation-log",
                    str(prep_path),
                    "--missingness-impact",
                    str(missingness_path),
                    "--model-spec",
                    str(approval_dir / "approved-model-specification.json"),
                    "--model-results",
                    str(result_dir / "model-results.csv"),
                    "--model-diagnostics",
                    str(result_dir / "model-diagnostics.csv"),
                    "--factor-tests",
                    str(result_dir / "factor-omnibus-tests.csv"),
                    "--category-support",
                    str(approval_dir / "category-support-screen.csv"),
                    "--shape-tests",
                    str(result_dir / "continuous-shape-tests.csv"),
                    "--influence-diagnostics",
                    str(result_dir / "influence-diagnostics.csv"),
                    "--iia-check",
                    str(result_dir / "iia-check.json"),
                    "--proportional-odds-check",
                    str(result_dir / "proportional-odds-check.json"),
                    "--count-dispersion-check",
                    str(result_dir / "count-dispersion-check.json"),
                    "--negative-binomial-need-check",
                    str(result_dir / "negative-binomial-need-check.json"),
                    "--zero-inflation-check",
                    str(result_dir / "zero-inflation-check.json"),
                    "--predictive-validation",
                    str(result_dir / "predictive-validation.json"),
                    "--model-summary",
                    str(result_dir / "model-summary.json"),
                    "--analysis-run-log",
                    str(result_dir / "analysis-run-log.json"),
                    "--figures-dir",
                    str(result_dir / "figures"),
                    "--cleaned-data",
                    str(data_path),
                    "--analysis-code",
                    str(RUN),
                    "--approval-record",
                    str(report_approval),
                    "--output-dir",
                    str(report_dir),
                )
                html = (report_dir / "final-report.html").read_text(encoding="utf-8")
                model_summary = json.loads((result_dir / "model-summary.json").read_text(encoding="utf-8"))
                manifest = json.loads((report_dir / "report-manifest.json").read_text(encoding="utf-8"))
                if "final_report_generated" in model_summary:
                    raise AssertionError("Model summary must not duplicate report-generation state")
                if manifest.get("final_report_generated") is not True:
                    raise AssertionError("Report manifest must be the completed report-state authority")
                for required in (
                    "统计检验小课堂",
                    "position: sticky",
                    "grid-template-columns: var(--toc-width) minmax(0, 1fr)",
                    "window.print()",
                    "max-height: var(--figure-max-height)",
                ):
                    if required not in html:
                        raise AssertionError(
                            f"Report check failed for {model_type}: {required}"
                        )
            completed_models.append(model_type)

        separated_data = root / "separated-data.csv"
        separated_prep = work_root / "separated-preparation-log.json"
        separated_task = work_root / "task-separated-logistic.json"
        separated_approval = work_root / "approval-separated-logistic"
        separated_results = work_root / "results-separated-logistic"
        pd.DataFrame(
            {
                "x": np.r_[np.linspace(-3, -0.1, 45), np.linspace(0.1, 3, 45)],
                "y": ["no"] * 45 + ["yes"] * 45,
            }
        ).to_csv(separated_data, index=False, encoding="utf-8-sig")
        separated_prep.write_text(
            json.dumps(
                {
                    "data_preparation_executed": True,
                    "modeling_executed": False,
                    **no_missingness_contract(),
                    "analysis_metadata": {
                        "analysis_exclusions": [],
                        "categorical_columns": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        separated_task.write_text(
            json.dumps(
                {
                    "status": "approved",
                    "approval": {"approval_id": "smoke-separated-task"},
                    "title": "separation gate",
                    "goal": "association",
                    "research_question": "x 与 y 是否有关？",
                    "outcome": "y",
                    "predictors": ["x"],
                    "controls": [],
                    "variables": [
                        {"column": "y", "inferred_type": "categorical"},
                        {"column": "x", "inferred_type": "numeric"},
                    ],
                    "variable_metadata": {},
                }
            ),
            encoding="utf-8",
        )
        separated_input = separated_approval / "workflow-decision-input.json"
        separated_input.parent.mkdir(parents=True, exist_ok=True)
        separated_input.write_text(
            json.dumps(
                {
                    "observation_structure": "independent",
                    "outcome_process": "standard",
                    "sampling_design": "simple",
                    "count_exposure": "not-applicable",
                }
            ),
            encoding="utf-8",
        )
        separated_workflow_approval = write_approval(
            separated_approval / "workflow-approval.json",
            "approve-workflow-support",
            separated_input,
        )
        invoke(
            str(ASSESS),
            "--task",
            str(separated_task),
            "--decision-input",
            str(separated_input),
            "--approval-record",
            str(separated_workflow_approval),
            "--output-dir",
            str(separated_approval),
        )
        invoke(
            str(APPROVE),
            "--task",
            str(separated_task),
            "--workflow-support",
            str(separated_approval / "workflow-support-assessment.json"),
            "--preparation-log",
            str(separated_prep),
            "--data",
            str(separated_data),
            "--model-type",
            "logistic",
            "--positive-class",
            "yes",
            "--output-dir",
            str(separated_approval),
        )
        separated_proposal = separated_approval / "model-specification-proposal.json"
        separated_model_approval = write_approval(
            separated_approval / "model-approval.json",
            "approve-model-specification",
            separated_proposal,
        )
        invoke_expect_failure(
            str(FINALIZE),
            "--proposal",
            str(separated_proposal),
            "--approval-record",
            str(separated_model_approval),
            "--task",
            str(separated_task),
            "--workflow-support",
            str(separated_approval / "workflow-support-assessment.json"),
            "--preparation-log",
            str(separated_prep),
            "--data",
            str(separated_data),
            "--output-dir",
            str(separated_approval),
            expected="separation",
        )
        unexpected_root_entries = sorted(
            path.name
            for path in root.iterdir()
            if path.name not in {"cleaned-data.csv", "separated-data.csv", "conversations"}
        )
        if unexpected_root_entries:
            raise AssertionError(
                f"Conversation artifacts leaked to workspace root: {unexpected_root_entries}"
            )
        print(
            json.dumps(
                {"ok": True, "models": completed_models, "rows": n},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
