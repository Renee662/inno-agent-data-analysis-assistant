---
name: execute-data-preparation
description: Apply an explicitly approved tabular-data preparation plan to a copy of a CSV, TSV, XLSX, or XLS source table, create cleaned-data.csv and auditable preparation logs, and verify that the raw file remains unchanged. Use only after confirm-data-preparation has created an approved-data-preparation-plan.json and before statistical modeling or final reporting.
---

# Execute Data Preparation

Execute only decisions that the user already approved. Preserve the raw table, record every action, and stop before modeling.

## Preconditions

Verify that the plan has all of the following:

- `status: approved`;
- `requires_user_confirmation: false`;
- `raw_data_unchanged: true`;
- `data_preparation_executed: false`;
- an `approved_decisions` array whose items, when present, have `status: user-confirmed` and `selected_option`; an empty array is valid after the user approves a plan that needs no discretionary changes.
- a confirmed `table_read_spec` whose source hash, sheet, header rows, data-start row, and column names still match the uploaded file.
- a `missingness_bias_screen` and conclusion-scope contract generated from that same confirmed source table.

If any condition fails, stop and return to the confirmation stage. Do not repair or infer approval.

## Run

Use the exact uploaded table and approved plan paths:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\execute-data-preparation\scripts\execute_preparation.py" `
  --input ".\uploads\source-table.csv" `
  --plan (Join-Path $run "work\data-preparation-confirmation\approved-data-preparation-plan.json") `
  --output-dir (Join-Path $run "work\data-preparation-execution")
```

For an Excel workbook, add `--sheet "Sheet1"` when the approved plan does not already identify the sheet.

The script must read CSV, TSV, XLSX, and XLS through the approved `table_read_spec`. Never fall back to the library's default first-row header behavior.

## Supported choices

- Duplicate rows: `keep`, `drop-exact-duplicates`, or `deduplicate-with-key` with approved `execution_parameters.key_columns`.
- Missing values: `complete-case`, `median-imputation`, `most-frequent`, `explicit-missing-category`, `missing-indicator`, or `model-specific-method` (recorded as deferred). Refuse `multiple-imputation` because it requires a separate model-compatible workflow rather than one cleaned CSV.
- Candidate outliers: `keep-and-run-diagnostics`, `transform` with approved parameters, `winsorize` with approved quantiles, or `exclude-with-domain-rule` with approved bounds.
- Categorical fields: `treat-as-categorical`, `keep-as-text-and-exclude`, or `manual-recode` with an approved mapping.
- Mixed types: `inspect-and-recode` or `coerce-invalid-to-missing` requires an approved `execution_parameters.target_type` (`numeric`, `datetime`, `text`, or `categorical`); `inspect-and-recode` may also carry an approved mapping. `exclude` preserves the field but excludes it from modeling.
- Variable inclusion: `exclude`, or `include-with-justification` with a non-empty approved user note.

If required execution parameters are absent, stop and return to `confirm-data-preparation`. Never invent thresholds, keys, transformations, or recoding maps.

## Outputs

Create only these files under the requested output directory:

- `cleaned-data.csv`: UTF-8 with BOM for convenient opening in Excel;
- `data-preparation-log.json`: hashes, row and column counts, choices, affected source row numbers, deferred actions, and analysis metadata;
- `data-preparation-log.md`: human-readable processing summary.
- `missingness-impact.csv`: compact descriptive comparison of outcome composition in missing versus observed groups.

Report the output paths and a concise before/after summary. Then stop and use `ask_user_question` with choices to accept and proceed, inspect affected rows, or ask why the sample size changed. Do not treat an explanation request as permission to model.

## Guardrails

- Never write to or rename the source table.
- Hash the source before and after execution; fail if it changes.
- Match the source filename to the approved plan.
- Recompute the missingness screen before any transformation and fail if it differs from the approved plan. Record original selected-field missingness separately from missingness that remains after preparation.
- Never describe the screen as proof of MCAR, MAR, or MNAR. If no model-level sensitivity analysis was run, restrict conclusions to the analyzed sample and state that limitation.
- Execute only items in `approved_decisions`.
- Preserve source row numbers in logs without exposing cell values.
- Do not drop identifiers or sensitive fields unless the approved plan explicitly says to do so.
- Do not fit a model, calculate p-values, draw inferential conclusions, or create `final-report.html` in this skill.
- Do not call external APIs. Use the managed shared interpreter in `INNO_DATA_ANALYSIS_PYTHON` and ask before installing missing dependencies.
