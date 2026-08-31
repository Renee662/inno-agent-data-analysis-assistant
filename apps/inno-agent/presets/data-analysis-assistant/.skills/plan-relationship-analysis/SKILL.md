---
name: plan-relationship-analysis
description: Turn a user's natural-language question about one dependent variable and multiple independent variables into a validated, reviewable analysis task card. Use after tabular profiling when the user names an outcome, predictors, controls, or asks which method fits their question. Check variables against data-profile.json, distinguish association, prediction, description, and causal intent, recommend a candidate method, write a draft plan, and stop for explicit confirmation; do not clean data or fit a model.
---

# Plan a Relationship Analysis

## Workflow

1. Require a completed `$env:INNO_CONVERSATION_DIR/work/data-profile/data-profile.json` whose selected sheet has an `auto-confirmed` or `user-confirmed` `table_read_spec`. If structure confirmation is pending or the specification is absent, return to `tabular-data-profiler`; do not plan from provisional columns. Also use `dataset-context.json` from `tabular-data-profiler` when it exists.
2. Extract from the user's message, without inventing fields:
   - analysis goal: `association`, `prediction`, `description`, or `causal`;
   - dependent variable;
   - independent variables;
   - optional control variables;
   - optional sheet and unit of analysis.
   - a short plain-language description of what the dataset records;
   - the research question as an ordinary reader would ask it;
   - a Chinese display name, unit, and meaningful interpretation increment for every selected field.
3. If the goal or dependent variable is ambiguous, use `ask_user_question` before creating a plan. Frame the goal as a meaningful decision rather than a model name:
   - A. understand how several factors relate to the result;
   - B. predict the result as accurately as possible;
   - C. compare average outcomes across groups.
   Explain how the answer changes the analysis strategy. Do not ask the user to choose OLS, Logistic, or another named method before this decision is clear.
4. Run the bundled planner from the workspace root. Quote every column name:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\plan-relationship-analysis\scripts\build_analysis_plan.py" --profile (Join-Path $run "work\data-profile\data-profile.json") --context (Join-Path $run "work\data-profile\dataset-context.json") --goal association --decision-goal relationships --outcome "<dependent-variable>" --predictors "<x1>" "<x2>" --controls "<optional-control>" --output-dir (Join-Path $run "work\analysis-plan")
   ```

   Map the user's choice to `--decision-goal relationships|prediction|group-comparison`. Add `--sheet`, `--unit-of-analysis`, or `--title` only when known. Omit `--controls` when none are proposed. If `public-dataset-verification.json` exists, add `--public-verification (Join-Path $run "work\data-profile\public-dataset-verification.json")`; this carries only source-attributed candidates into semantic review and never confirms identity. After generation, add the user-reviewed `dataset_summary`, `research_question`, `report_title`, and `variable_metadata` fields to `analysis-task.json`; do not invent business meanings or units from column names alone.
5. Read `$env:INNO_CONVERSATION_DIR/work/analysis-plan/analysis-task.json` and `analysis-plan.md`.
6. Resolve `semantic_review` before asking for task-card approval:
   - Ask only about the selected outcome, predictors, and controls in `semantic_review.fields`. Never ask the user to explain ordinary unselected columns.
   - Skip a selected field only when the user has already explicitly supplied and confirmed its meaning for this dataset. Record that fact in `variable_metadata`; do not treat a familiar abbreviation as confirmation.
   - Follow `question_batches` and call `ask_user_question` for at most three fields at a time.
   - When `source_evidence` exists, summarize one concise candidate meaning and cite its relative local source in the option description. Label it as a candidate; the source does not automatically confirm the field.
   - For `public-dataset-candidate` evidence, show its clickable source URL, publisher, repository ID, version, and column-match status. Its `identity_confirmed: false` value is binding: do not describe the field meaning as source-confirmed until the user has resolved source/version conflicts.
   - When no local evidence exists, a meaning inferred from the abbreviation, value pattern, or statistical type must be labelled “Agent 候选推测”. Never present it as established fact.
   - Each field question should offer: `采用这个候选含义`, `这个含义不对，我来补充`, and a field-specific help choice such as `我不确定 attr 为什么可能表示吸引力评分`. The fill-in box is used for the user's correction or clarification.
   - A help, uncertainty, objection, or custom answer is not confirmation. Explain the evidence and consequence in ordinary language, update the candidate if needed, and ask that field again.
   - Do not ask for a unit when the field is categorical; ask for every category's meaning instead. For a numeric field, ask for its unit and a meaningful change used in interpretation.
7. Inspect `semantic_review.structural_candidates` only to clarify the unit of analysis and record dependence. These are not automatically controls. If the unit or record independence is uncertain, ask one structured question covering no more than the listed three candidates. Explain that repeated IDs, pairs, batches, waves, or dates can mean rows are not independent. Do not ask about other structural-looking columns unless later evidence makes them relevant.
8. Store each resolved selected field under `variable_metadata` with this minimum audit trail:

   ```json
   {
     "raw_column": {
       "display_name": "中文名称",
       "meaning": "该字段在本数据中的含义",
       "unit": "数值单位或空",
       "category_meanings": {},
       "interpretation_increment": 1,
       "interpretation_increment_label": "1小时",
       "semantic_status": "user-confirmed",
       "source_type": "user|companion-file|workbook-context-sheet|agent-hypothesis",
       "source_path": "相对路径或空",
       "user_confirmed": true
     }
   }
   ```

   Keep `analysis-task.json` in `draft` status while any selected field remains unresolved. Do not show the overall task-card approval question yet.
9. Present the generated task card as a draft. Clearly identify:
   - a two- or three-sentence dataset introduction for non-technical readers;
   - the research question without leading with dependent/independent-variable terminology;
   - the confirmed Chinese name and unit of each selected field;
   - goal and unit of analysis;
   - outcome, predictors, and controls;
   - inferred variable types and missingness;
   - identifier or sensitive-field risks;
   - the user's decision goal, recommended candidate method, why it answers that goal, and alternatives;
   - unresolved questions.
10. End with the standard stage summary: 已完成、主要发现、建议、待确认.
11. Only after all selected-field semantics and the analysis unit are resolved, use `ask_user_question` for task-card approval. Set `questionKind: task-card`, `documentPath` and `approvalArtifactPath` to the current conversation's `work/analysis-plan/analysis-task.json`, and `approvalAction: approve-analysis-task`. Mark exactly one option with `approvalDecision: approve`; mark revise/help choices explicitly. The current immutable proposal must render inside the questionnaire. A request for explanation is not approval: explain with this dataset, then ask the decision again.
12. After an approving response, use the approval-record path returned by `ask_user_question` to run `approve_analysis_task.py`. This creates `approved-analysis-task.json`; never change or replace the proposal after approval:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\plan-relationship-analysis\scripts\approve_analysis_task.py" `
     --proposal (Join-Path $run "work\analysis-plan\analysis-task.json") `
     --approval-record "<path returned by ask_user_question>" `
     --output-dir (Join-Path $run "work\analysis-plan")
   ```

13. Stop if no approval receipt was produced. Do not clean data, encode variables, choose a final model, or begin analysis from the draft task card.

## Decision Rules

- Treat “relationship”, “association”, and “correlation” as association unless the user says otherwise.
- Treat “predict”, “forecast”, or out-of-sample performance as prediction.
- Treat “cause”, “effect”, “impact”, or “导致/影响” as potentially causal and request design information; never silently downgrade causal language to ordinary regression.
- Treat method recommendations as candidates, not final decisions.
- A public-dataset name, familiar column abbreviation, local codebook, or value pattern can provide a candidate meaning but never replaces version, column, and user verification.
- Reject missing columns and role conflicts instead of guessing.
- Warn when an identifier, sensitive candidate, constant field, or high-missingness field is selected.
- Ask whether rows are independent or repeated/nested observations.
- Ask for missing variable names, units, category meanings, and meaningful changes such as one hour, ten percentage points, or one standard amount.
- Ask which controls are theoretically justified; do not select controls only because they improve significance.

## Guardrails

- Read only the profile and dataset-context artifacts; do not inspect or copy raw rows during planning.
- Preserve the confirmed `table_read_spec` in `analysis-task.json`; never reconstruct header settings from column names.
- Write only under `$env:INNO_CONVERSATION_DIR/work/analysis-plan/`.
- Do not modify the source table or previous profile outputs.
- Do not call external APIs.
- If planning fails, report the exact validation error and available column names; never invent a task card.

## Output Contract

- `analysis-task.json`: immutable machine-readable proposal shown to the user.
- `approved-analysis-task.json`: approved task plus question/session/artifact-hash receipt.
- `analysis-plan.md`: human-readable task card for user review.

Both outputs must have `status: draft` and require explicit user confirmation before later Skills may execute statistical analysis.

The confirmed JSON must retain these reader-facing fields for final reporting:

```json
{
  "report_title": "中文报告标题",
  "dataset_summary": "这是什么数据集、包含什么对象和信息",
  "research_question": "普通读者能够直接理解的问题",
  "variable_metadata": {
    "raw_column": {
      "display_name": "中文名称",
      "unit": "单位",
      "interpretation_increment": 1,
      "interpretation_increment_label": "1小时"
    }
  }
}
```
