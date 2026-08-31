---
name: plan-data-preparation
description: Create a reviewable data-preparation plan from a user-confirmed relationship-analysis task and its tabular profile. Use after the user approves the dependent variable, predictors, controls, analysis goal, sheet, and unit of analysis. Propose—but do not execute—decisions for missing values, duplicate rows, type conversion, categorical encoding, outliers, scaling, and variable inclusion. Write a draft plan with pending decisions and stop for explicit confirmation; never modify source data or create cleaned-data.csv.
---

# Plan Data Preparation

## Workflow

1. Require both a completed `data-profile.json` and an analysis task card produced by `plan-relationship-analysis`. Their confirmed `table_read_spec` values must match exactly; otherwise stop and regenerate the task card.
2. Confirm in conversation that the user has approved the task card or specified revisions. Do not treat silence as approval.
3. Run the planner only from `approved-analysis-task.json`. The planner rejects a draft or a task without a questionnaire receipt:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\plan-data-preparation\scripts\build_preparation_plan.py" --input ".\uploads\source-table.csv" --profile (Join-Path $run "work\data-profile\data-profile.json") --task (Join-Path $run "work\analysis-plan\approved-analysis-task.json") --output-dir (Join-Path $run "work\data-preparation")
   ```

4. Read `$env:INNO_CONVERSATION_DIR/work/data-preparation/data-preparation-plan.json` and `data-preparation-plan.md`.
5. Require `execution_contract.status: compatible` before presenting any question. If the planner reports an unsupported decision ID or option, stop with a concise capability error; do not ask the user to approve a choice the executor cannot run.
6. Present each pending decision with a recommendation, alternatives, and consequences. Cover:
   - exact duplicate rows and fully empty rows;
   - missing outcome, predictor, and control values;
   - mixed or incorrect types;
   - categorical encoding and reference levels;
   - candidate outliers;
   - scaling or transformation;
   - constant, identifier, or sensitive fields.
   Before asking, show the missingness sample-composition screen: rows retained/lost under complete-case handling and the outcome distribution in retained versus excluded rows. Explain that this is descriptive evidence only and cannot identify MCAR, MAR, or MNAR.
7. Do not turn a recommendation into a decision. Use `ask_user_question` to approve or revise every item marked `pending`, grouping at most three related decisions in one call. Each technical decision must include an option to learn how it changes the sample or result.
8. End with the standard stage summary: 已完成、主要发现、建议、待确认.
9. Stop. Do not read raw rows, write transformed data, or start modeling in this Skill.

## Decision Principles

- Never impute a missing outcome by default.
- Never claim that missingness is random from a non-significant comparison or a small distribution difference.
- When selected variables contain missingness, restrict conclusions to the analyzed sample unless a separate, approved model-estimate sensitivity workflow has actually been completed. A single imputed CSV is not such a workflow.
- Never delete an outlier solely because it crosses an automatic IQR threshold.
- Never encode a numeric-looking category as continuous without semantic confirmation.
- Do not standardize by default for explanatory analysis; offer standardized effects separately when useful.
- For prediction, recommend fitting imputers, encoders, and scalers on training data only to prevent leakage.
- Recommend excluding constants and identifiers, but require confirmation when the user explicitly selected them.
- Preserve sensitive variables only when their analytical necessity and privacy treatment are confirmed.
- Preserve the original file and record every approved transformation for later reporting.

## Guardrails

- Read the confirmed source table only to reproduce the table-read contract and calculate the missingness sample-composition screen; do not expose raw rows.
- Write only under `$env:INNO_CONVERSATION_DIR/work/data-preparation/`.
- Keep `status: draft` and `requires_user_confirmation: true`.
- Do not create `cleaned-data.csv`; a later execution Skill will do that only from an approved plan.
- Do not call external APIs.
- If profile and task variables disagree, stop with a validation error instead of guessing.
- Carry the confirmed `table_read_spec` into the draft plan so execution cannot silently fall back to a first-row header.

## Output Contract

- `data-preparation-plan.json`: structured plan containing row-level actions, per-variable actions, warnings, pending decisions, and confirmation state.
- `data-preparation-plan.md`: human-readable plan for user review.
- embedded `missingness_bias_screen`: retained/lost sample counts and descriptive outcome comparisons, bound to the source-table hash through `table_read_spec`.

These outputs are planning artifacts. They do not authorize data changes until the user explicitly confirms the pending decisions.
