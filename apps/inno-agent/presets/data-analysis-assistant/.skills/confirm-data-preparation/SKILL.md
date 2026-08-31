---
name: confirm-data-preparation
description: Review a draft tabular-data preparation plan with the user, collect one explicit final choice for every pending cleaning decision, and create an immutable confirmed decision record without asking for a second whole-plan approval. Use after plan-data-preparation has produced data-preparation-plan.json and before any cleaning, transformation, modeling, or cleaned-data.csv creation.
---

# Confirm Data Preparation

Turn a draft preparation plan into a confirmed executable plan. Keep each choice informed and reversible before submission: explain choices, record the user's decisions, and stop before modifying data. A submitted structured choice is final for that item; do not ask the user to approve the assembled plan again.

## Workflow

1. Read the draft `data-preparation-plan.json` and verify:
   - `status` is `draft`;
   - `requires_user_confirmation` is `true`;
   - `raw_data_unchanged` is `true`.
2. Present each item in `pending_decisions` in plain language and collect the choice through `ask_user_question` (at most three related items per call). For every item:
   - state what was detected;
   - explain the recommended choice and its consequence;
   - list the allowed options;
   - provide a specific “先解释这会怎样影响样本或结果” option and allow a written objection;
   - ask the user to choose with buttons instead of typing option numbers.
   Before the first missing-value choice, summarize `missingness_bias_screen`: how many rows complete-case handling would retain and exclude, and how the outcome composition differs. In one or two sentences state that these are descriptive comparisons, not proof that missingness is random and not a substitute for comparing model estimates across missing-data methods.
3. Do not infer a choice from an ambiguous answer. A help or uncertainty choice is not approval. Answer the clarification with evidence from this plan, then present the same structured decision again without advancing the workflow.
4. If the user says “use all recommendations”, still record the exact choice for every decision through structured questions; do not replace item-level choices with a single blanket approval. If there are no pending decisions, write an empty `decisions` object without asking an overall approval question.
5. Treat each submitted non-help answer as the final choice for that item. If a choice needs keys, thresholds, bounds, a transform, or a recoding map, collect those exact values as `execution_parameters` in that item's interaction before accepting it. Never invent them. Once every item is resolved, do not call `ask_user_question` for a whole-plan approval, revision, or confirmation.
6. Create a UTF-8 JSON decision file in the confirmation output directory directly from the completed item-level answers:

```json
{
  "decisions": {
    "rows:duplicates": {
      "choice": "keep",
      "note": "Optional user rationale",
      "execution_parameters": {}
    }
  }
}
```

7. Run:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\confirm-data-preparation\scripts\approve_preparation_plan.py" `
  --draft (Join-Path $run "work\data-preparation\data-preparation-plan.json") `
  --decisions (Join-Path $run "work\data-preparation-confirmation\user-decisions.json") `
  --output-dir (Join-Path $run "work\data-preparation-confirmation")
```

8. Show the assembled execution plan and output paths in ordinary conversation text, without another questionnaire. Then return control so `.skills/execute-data-preparation` can run immediately. Do not clean, transform, encode, impute, remove rows, fit a model, or create `cleaned-data.csv` inside this skill.

## Guardrails

- Require a choice for every pending decision; reject missing or unknown decision IDs.
- Permit an empty decision set only when the draft contains no pending decisions; no redundant overall approval is required.
- Accept only choices listed in each decision's `options` array.
- Record the decision-file hash and `confirmation_method: structured-item-questionnaire`; do not claim that a separate whole-plan approval receipt exists.
- Preserve the draft plan and source data. Write only the approved plan and approval record to the requested output directory.
- Keep `raw_data_unchanged: true` in approved outputs.
- Never treat an IQR candidate as proof that a row should be removed.
- Never impute an outcome automatically.
- If selected fields contain missingness, tell the user that the bundled single-dataset workflow limits conclusions to the analyzed sample. Do not say that median/mode imputation “solves” selection bias or that a missing-indicator proves unbiasedness.
- If the user needs conclusions that are robust to alternative missing-data assumptions, stop and route to a separately approved model-level sensitivity or multiple-imputation workflow; do not represent one cleaned CSV as multiple imputation.
- Never interpret statistical association as causation.
- Do not call an external API or install a package. If a later step needs one, explain cost and ask before use.

## Outputs

The script creates:

- `approved-data-preparation-plan.json`: machine-readable confirmed plan with source-plan hash and item-level decision record.
- `approved-data-preparation-plan.md`: human-readable execution-plan summary.

Both outputs must state that data preparation has not yet been executed.
