---
name: run-statistical-analysis
description: Confirm a statistical model specification and run an auditable OLS, binary logistic, Poisson, negative-binomial, multinomial-logistic, or ordinal-logistic regression on cleaned tabular data, producing coefficient tables, effect intervals, diagnostics, learning prompts, plots, and run logs. Use after execute-data-preparation has produced cleaned-data.csv and data-preparation-log.json, and before creating final-report.html.
---

# Run Statistical Analysis

Confirm the exact model before fitting it. Report effect direction, magnitude, uncertainty, and diagnostics without treating association as causation.

## Step 0: Enforce workflow fit before discussing a model

Read the analysis task and ask only for unresolved structure facts that can change the method:

- observation structure: independent rows, repeated measures, nested/clustered units, paired records, or time series;
- outcome process: ordinary outcome, time-to-event, censored, or structurally zero-inflated;
- sampling design: simple, weighted, clustered, stratified, or complex survey;
- for a count outcome, whether counts require an exposure, population, or observation-time offset.

Use `ask_user_question` in batches of at most three, always with `questionKind: decision`. Each question must include a data-specific uncertainty/help option. Help or `unknown` is not approval and cannot be silently converted to `independent`, `standard`, or `simple`. Put the final proposed values in `work/statistical-analysis/workflow-decision-input.json`, show that file in a final questionnaire, set `approvalAction: approve-workflow-support`, and mark exactly one option `approvalDecision: approve`. After the tool returns an approval-record path, run:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\run-statistical-analysis\scripts\assess_workflow_support.py" `
  --task (Join-Path $run "work\analysis-plan\approved-analysis-task.json") `
  --decision-input (Join-Path $run "work\statistical-analysis\workflow-decision-input.json") `
  --approval-record "<path returned by ask_user_question>" `
  --output-dir (Join-Path $run "work\statistical-analysis")
```

The decision-input JSON must contain `observation_structure`, `outcome_process`, `sampling_design`, and `count_exposure` using the values documented above. Read `workflow-support-assessment.json`:

- `supported`: continue to Step 1;
- `needs-user-information`: stop and ask only about `unknown_dimensions`, then regenerate the assessment;
- `specialized-workflow-required`: stop the bundled modeling workflow. Explain each `blocking_reasons` item using the current dataset, state the required specialized workflow, and do not run either approval or analysis scripts.

The following always require another workflow: repeated/paired observations, multilevel nesting, time series, survival time or censoring, zero-inflated/hurdle processes, complex sampling or survey weights, count offsets/exposure, and causal identification. Prediction is allowed only through the bundled fixed-specification deterministic five-fold sample-out validation; it is internal validation, not evidence of performance in another population. A generic user instruction to “continue anyway” never overrides `execution_allowed: false`.

## Step 1: Confirm the model specification

Read the analysis task, preparation log, and cleaned table. Present all of the following to the user:

- outcome;
- predictors;
- controls;
- categorical variables;
- goal: association, prediction, description, or causal;
- the user's plain-language decision goal before any model name: understand factor relationships, maximize prediction, or compare group differences;
- proposed model: `ols`, `logistic`, `poisson`, `negative-binomial`, `multinomial-logistic`, or `ordinal-logistic`;
- robust standard-error choice and confidence level;
- for every categorical predictor/control, its proposed reference category and whether it came from approved task metadata, an explicit choice, or the automatic most-frequent-category default;
- sample-size or model-identification warnings.

Before showing the model proposal, inspect `collinearity_screen`. If it is `requires-review` or `revision-required`, introduce the decision with no more than these two plain sentences: “共线性是多个自变量包含大量重复信息。VIF越高，越难稳定拆分每个变量的独立作用。” For `revision-required`, deterministic duplicate encodings make the design unidentified: offer only revising the overlapping variables or requesting a short explanation, and do not finalize the proposal. For `requires-review`, ask one decision question with these substantive choices: revise the overlapping variable set before fitting; retain the approved set but forbid stable individual-effect claims for affected terms; or request a short explanation. Never silently delete a variable or turn the help choice into approval.

For Logistic, multinomial-Logistic, and ordinal-Logistic models, inspect `category_support_screen` before offering model approval. Introduce any issue in no more than these two plain sentences: “稀疏类别是某个类别样本或某种结果太少。分类分离是某个类别几乎或完全只对应一种结果，会让普通Logistic系数和优势比失控。” Show `category-support-screen.csv`, including every retained factor level and its count in every outcome category. Treat zero outcome cells that imply binary/multinomial separation, or an ordinal level confined to one outcome, as mathematical blockers. Treat small observed/expected cells and low observations-per-parameter as combined planning evidence, not universal scientific cutoffs; explain that the recorded 5-cell and 10-per-parameter values are warning benchmarks only. If status is `requires-review` or `revision-required`, do not offer ordinary model approval. Offer revising preparation to merge substantively compatible levels, revising the task to exclude the affected factor with a rationale, providing more data, or stopping/requesting a short explanation. Every merge map or exclusion must be separately approved in the preparation/task artifact. Never merge, drop, relabel, or continue automatically.

For OLS, binary Logistic, Poisson, negative-binomial, multinomial-Logistic, and ordinal-Logistic regression, determine continuous-variable forms automatically instead of asking the user to choose statistical terminology. Apply this fixed policy before fitting: use an approved metadata form only when it includes a substantive rationale; otherwise use a four-knot restricted cubic spline when the variable has at least eight distinct values, and use a linear form with an explicit “nonlinearity not adequately evaluable” limitation when it does not. Use a quadratic form only for an already approved U-shaped or inverted-U-shaped substantive hypothesis. Never fit several forms and select the smallest p-value, AIC, or most favorable conclusion. Show the compact model-appropriate binned preview and the proposed form, source, rationale, parameter cost, and limitation in the model proposal. For multinomial outcomes, preview each unordered category's proportion; for ordinal outcomes, preview each approved level's proportion. Never preview or model a category code as though it were a continuous outcome. Ask the user only to approve or revise the plain-language overall proposal; do not ask them “linear, quadratic, or spline?”.

For multinomial Logistic regression, inspect `iia_check` before offering approval. Explain in no more than two plain sentences: “IIA假设是：移除一个结果类别后，其余类别之间的相对关系不应明显改变。若检查发现明显变化，普通多分类Logistic可能把天然相似或嵌套的类别关系简化过度。” Use the deterministic generalized Hausman-McFadden category-deletion screen, preserve every deletion result, and apply Holm adjustment across deletions. `sensitivity-detected` or `not-evaluated` blocks ordinary multinomial Logistic approval; offer a specialized nested/multinomial-probit choice workflow, a substantively justified revision of compatible outcome categories, or stopping for an explanation. `clear-no-detected-sensitivity` permits approval only with the limitation that the screen does not prove IIA and domain knowledge about nested alternatives still matters.

For ordinal Logistic regression, inspect `proportional_odds_check` before offering approval. Explain in no more than two plain sentences: “有序Logistic假设同一因素对每个等级分界的影响大致相同。检查不通过时，一个统一系数会把不同分界的影响过度合并。” The deterministic check fits threshold-specific cumulative logits and uses a row-cluster-robust Wald test of equal slopes at the approved alpha level. `violation-detected` or `not-evaluated` blocks ordinary ordinal Logistic approval; offer multinomial Logistic, a specialized partial-proportional-odds workflow, or stopping for an explanation. `clear-no-detected-violation` permits approval but must retain the limitation that absence of detected evidence does not prove the assumption.

For ordinary Poisson regression, inspect `count_dispersion_check` before offering approval. Explain in no more than two plain sentences: “Poisson假设在其他因素相同时，计数的波动与平均数量大致匹配。若实际波动系统性更大，普通Poisson可能把不确定性写得过小。” Use the one-sided adjusted NB2 auxiliary score screen at the approved alpha level; keep Pearson dispersion descriptive and never use a fixed heuristic cutoff to switch models automatically. `overdispersion-detected` or `not-evaluated` blocks ordinary Poisson approval; offer returning to model choice for negative-binomial regression or stopping for an explanation. `clear-no-detected-overdispersion` permits approval but does not prove the variance assumption.

For negative-binomial regression, inspect `negative_binomial_need_check` before offering approval. Explain in no more than two plain sentences: “负二项回归比Poisson多放宽了一层计数波动。只有数据明确显示这种额外波动时，才有必要使用它。” Use the same adjusted NB2 auxiliary evidence used by the Poisson gate. `no-detected-need-for-extra-dispersion` or `not-evaluated` blocks negative-binomial approval; return to the ordinary-Poisson choice or stop for an explanation. Never select negative-binomial merely because it is more flexible.

For both Poisson and negative-binomial regression, inspect `zero_inflation_check` before approval. Explain in no more than two plain sentences: “零过多是指数据中的0明显多于当前计数模型所能解释的数量。这可能意味着一部分对象根本不会发生该事件，需要零膨胀或hurdle模型。” Compare the observed zero count with the fitted model's expected zero count using the registered one-sided screen. `excess-zeros-detected` or `not-evaluated` blocks the ordinary count model and requires a specialized zero-inflated/hurdle workflow or stopping; `clear-no-detected-excess-zeros` permits fitting but does not prove that no structural-zero process exists.

Use these model boundaries:

- `ols`: approximately continuous numeric outcome;
- `logistic`: exactly two outcome categories, with an explicitly confirmed positive category;
- `poisson`: non-negative integer count outcome;
- `negative-binomial`: non-negative integer count outcome whose variance materially exceeds its mean;
- `multinomial-logistic`: at least three unordered categories, with a user-confirmed reference category;
- `ordinal-logistic`: at least three ordered categories, with every level placed in a user-confirmed order;
- refuse mixed-effects, time-series, survival, or causal models until a dedicated workflow exists.

Do not begin with “Do you agree to use OLS?”. Use `ask_user_question` with `questionKind: decision` to ask what real question the user wants to answer and collect exact settings. A help choice is not approval. After all settings are resolved, run the proposal builder below; it does not approve or fit anything:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\run-statistical-analysis\scripts\approve_model_spec.py" `
  --task (Join-Path $run "work\analysis-plan\approved-analysis-task.json") `
  --workflow-support (Join-Path $run "work\statistical-analysis\workflow-support-assessment.json") `
  --preparation-log (Join-Path $run "work\data-preparation-execution\data-preparation-log.json") `
  --data (Join-Path $run "work\data-preparation-execution\cleaned-data.csv") `
  --model-type ols `
  --robust-se HC3 `
  --confidence-level 0.95 `
  --output-dir (Join-Path $run "work\statistical-analysis")
```

For logistic regression, add `--positive-class "confirmed value"`.
For OLS, Logistic, Poisson, negative-binomial, multinomial-Logistic, and ordinal-Logistic regression, normally omit `--continuous-form`; the proposal builder applies the fixed automatic policy and records the exact centering, scaling, transformed-term map, decision provenance, four spline knots at the 5th/35th/65th/95th percentiles when selected, and the model-appropriate unadjusted binned preview. Use `--continuous-form` only to carry an already approved domain-based override with a recorded rationale, never as a technical question posed to the user.
For multinomial logistic regression, add `--reference-class "confirmed reference value"`.
For ordinal logistic regression, add `--category-order "lowest" "middle" "highest"` with every observed category exactly once.
For categorical predictors or controls, add one `--categorical-reference "column=confirmed value"` per variable when the user changes the proposed default. If omitted, the proposal builder first uses the approved task metadata and otherwise proposes the most frequent observed category. The proposal must still display every actual reference category, and the final approval covers those choices.

The builder must finish successfully and `design_matrix_preflight.status` must be `passed` before any model-approval card is shown. A failed preflight writes `design-matrix-preflight.json` and means the data structure, category encoding, functional forms, or variable set must be revised first; never ask the user to approve that draft. Any change to the task, preparation log, cleaned data, category references, or functional forms invalidates the old proposal. Regenerate it silently after the revision and show exactly one model-specification approval card only when all structural and model-specific gates pass.

Show `model-specification-proposal.json` in the single final questionnaire with `questionKind: task-card`, `documentPath` and `approvalArtifactPath` pointing to that proposal, and `approvalAction: approve-model-specification`. Mark exactly one option `approvalDecision: approve`. When `collinearity_screen.status` is `requires-review`, the approve option must explicitly say that the overlapping variables are retained and affected single coefficients will not be treated as stable independent effects; the revise option returns to variable selection. When `category_support_screen.status` is `requires-review` or `revision-required`, do not include an approve option: show the category-count artifact and return to preparation/task revision, more-data collection, or stopping. After an approving response, finalize the immutable proposal:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\run-statistical-analysis\scripts\finalize_model_spec.py" `
  --proposal (Join-Path $run "work\statistical-analysis\model-specification-proposal.json") `
  --approval-record "<path returned by ask_user_question>" `
  --task (Join-Path $run "work\analysis-plan\approved-analysis-task.json") `
  --workflow-support (Join-Path $run "work\statistical-analysis\workflow-support-assessment.json") `
  --preparation-log (Join-Path $run "work\data-preparation-execution\data-preparation-log.json") `
  --data (Join-Path $run "work\data-preparation-execution\cleaned-data.csv") `
  --output-dir (Join-Path $run "work\statistical-analysis")
```

When an ordinal proposal has `proportional_odds_check.status` of `violation-detected` or `not-evaluated`, do not include an approve option. Show `proportional-odds-check.json`, explain the affected grade boundaries in plain language, and return to model choice. Do not let a generic “continue anyway” override this gate.

When a multinomial proposal has `iia_check.status` of `sensitivity-detected` or `not-evaluated`, do not include an approve option. Show `iia-check.json`, explain which deleted category changed the remaining comparisons, and return to model or outcome-structure revision. Do not let a generic “continue anyway” override this gate or automatically merge categories.

When a Poisson proposal has `count_dispersion_check.status` of `overdispersion-detected` or `not-evaluated`, do not include an approve option. Show `count-dispersion-check.json`, explain that the count variability exceeds the ordinary Poisson assumption after accounting for the proposed factors, and return to model choice for an explicitly approved negative-binomial model or stop. Never switch models automatically.

When a negative-binomial proposal has `negative_binomial_need_check.status` other than `extra-dispersion-supported`, do not include an approve option. Show `negative-binomial-need-check.json`, explain that the data have not established the extra variation that justifies the more flexible model, and return to Poisson or stop. When either count model has `zero_inflation_check.status` other than `clear-no-detected-excess-zeros`, do not include an approve option. Show `zero-inflation-check.json` and route detected excess zeros to a specialized zero-inflated/hurdle workflow; never ignore the gate or automatically fit a new model.

## Step 2: Fit and diagnose

Only after `approved-model-specification.json` exists, run:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\run-statistical-analysis\scripts\run_analysis.py" `
  --data (Join-Path $run "work\data-preparation-execution\cleaned-data.csv") `
  --spec (Join-Path $run "work\statistical-analysis\approved-model-specification.json") `
  --preparation-log (Join-Path $run "work\data-preparation-execution\data-preparation-log.json") `
  --output-dir (Join-Path $run "work\statistical-analysis\results")
```

## Outputs

Create:

- `workflow-support-assessment.json` and `.md`;
- `approved-model-specification.json` and `.md`;
- `category-support-screen.csv` with every retained categorical level and outcome count;
- `results/model-results.csv`;
- `results/model-diagnostics.csv`;
- `results/factor-omnibus-tests.csv`;
- `results/continuous-shape-tests.csv`;
- `results/influence-diagnostics.csv` with model-specific candidate evidence and at most five highest-priority one-row-deletion refits;
- `results/iia-check.json` recording the multinomial IIA sensitivity gate or `not-applicable` for other models;
- `results/proportional-odds-check.json` recording the ordinal assumption gate or `not-applicable` for other models;
- `results/count-dispersion-check.json` recording the ordinary-Poisson overdispersion gate or `not-applicable` for other models;
- `results/negative-binomial-need-check.json` recording whether extra dispersion supports negative-binomial regression or `not-applicable` for other models;
- `results/zero-inflation-check.json` recording the excess-zero gate for Poisson/negative-binomial models or `not-applicable` otherwise;
- `results/predictive-validation.json` recording deterministic five-fold internal sample-out validation for prediction goals or `not-applicable` otherwise;
- `results/model-summary.json`;
- `results/analysis-run-log.json`;
- `results/analysis-summary.md`;
- `results/learning-prompts.json`;
- diagnostic PNG files under `results/figures/`.

Use a consistent academic-paper figure style by default: white background, muted blue-gray palette, Chinese titles and axis labels, Microsoft YaHei with Chinese fallback fonts, bold titles, comfortably large text, sparse light grid lines, and 300 DPI PNG output. Preserve honest axes, uncertainty intervals, outliers, and diagnostic reference lines; visual polish must not hide evidence.

After the script completes, do not reveal the full model answer immediately. Read `learning-prompts.json` and run this sequence:

1. If `sequence` starts with `collinearity-review`, introduce it with the one-sentence explanation stored in that prompt, ask its decision question, and stop. A revise choice returns to Step 1; a retain choice keeps the model but does not remove the interpretation restriction. If the sequence contains `influence-review`, explain in one or two sentences that an outlier looks unusual while a high-influence record materially changes the model, then show the current candidate and case-deletion evidence. Never treat a candidate flag as permission to delete a row.
2. Ask the observation question through `ask_user_question`, including a “我看不出来” option. Set `questionKind` to `observation`, and set `imagePath`, `imageAlt`, and `imageCaption` from `observation.workspace_image_path`, `image_alt`, and `image_caption`, so the reference plot appears as a compact image inside the conversation. If there is no valid image, do not ask the user to judge a plot. Do not send the user to the right preview pane. Stop for the user's answer.
3. Give evidence feedback from the same JSON, then ask the diagnostic-reasoning question through `ask_user_question`, including a concept-specific “我不懂这项诊断” option. When the question asks the user to inspect a diagnostic plot, set `questionKind` to `observation` and attach `diagnostic_reasoning.workspace_image_path` with the same three question-image fields; without a valid image, ask only a numerical/evidence interpretation question. Stop again.
4. Give evidence feedback tied to the current diagnostic values, then present the complete model results and ask for review before final report generation.

These are low-stakes reasoning prompts, not a test and not a substitute for model-specification approval.

For an unordered multinomial outcome, the observation prompt must use category proportions, grouped distributions, and an omnibus association test. Never assign ordinal meaning to category codes or ask whether an unordered outcome is positively or negatively correlated with a predictor.

## Guardrails

- Require `data_preparation_executed: true` and `modeling_executed: false` in the preparation log.
- Require the approved missingness-impact and conclusion-scope contract. If model-estimate sensitivity was not completed, carry an analyzed-sample-only restriction into the model summary and record any additional cleaned-data rows excluded for missingness separately.
- Require a task-matching workflow assessment with `status: supported`, `execution_allowed: true`, and user-confirmed structure answers. Never infer missing structure values merely to pass the gate.
- Require a questionnaire approval receipt matching the exact immutable model proposal before writing the approved model specification. A command-line boolean is never approval.
- Refuse an unidentified, rank-deficient, singular, or nonconvergent model. Treat convergence, Hessian-inversion, perfect-separation, and singular-matrix warnings as failures rather than reportable results.
- Stop before writing results when any estimate, standard error, confidence bound, covariance element, or fitted probability is nonfinite.
- Recompute the exact approved category-support screen immediately before fitting. Treat separation as a mathematical blocker and sparse cells/low observations-per-parameter as combined review evidence rather than a universal threshold. Refuse unresolved classification support risk instead of accepting unstable extreme coefficients.
- Do not silently remove predictors, controls, or categories.
- Treat VIF at or above 10 as a severe individual-coefficient interpretation restriction, not a decorative warning. Preserve the fitted model for audit, mark affected terms in the result artifact, explain the restriction before report approval, and require a separately approved revised specification for any alternative-variable sensitivity fit.
- For OLS, binary Logistic, Poisson, negative-binomial, multinomial-Logistic, and ordinal-Logistic regression, require the deterministic functional-form policy and provenance record for every continuous predictor/control. Treat quadratic and spline basis coefficients as a joint representation: test the overall variable and nonlinear component jointly, never interpret a basis coefficient or its transformed value as a real-world one-unit effect, and report a model-appropriate adjusted prediction curve with a confidence band. Multinomial Logistic must show a separate adjusted probability curve and confidence band for every unordered category; ordinal Logistic must do the same for every approved ordered level. Never reduce these outcomes to a numeric-code trend. High VIF internal to an approved nonlinear basis is structural and must not be mistaken for collinearity between separate predictors; keep those basis terms out of the individual-VIF conclusion while retaining their VIF values for audit.
- Require the approved reference-category map to cover every categorical predictor/control exactly. Fit the design matrix with those categories first, record them in model outputs, and stop if the encoded or reported references differ.
- Report complete-case rows used by the model and cleaned-data row numbers excluded for remaining missingness.
- Use model-specific influence evidence: leverage, standardized residuals, and Cook distance where defined; for multinomial and ordinal models use design leverage plus observed-class deviance surprise and case-deletion refits without claiming a Cook distance. Distinguish `not-evaluated` from zero candidates. Retain the original fit, never delete a candidate automatically, and compare one-row-deletion refits for at most the five highest-priority candidates.
- For ordinal Logistic, require `proportional_odds_check.model_fitting_allowed: true` before approval and fitting. Recompute the cluster-robust equal-slopes Wald screen immediately before the final fit and stop if its material evidence differs from the approved proposal. Never describe a non-significant check as proof that proportional odds holds.
- For multinomial Logistic, require `iia_check.model_fitting_allowed: true` before approval and fitting. Recompute the generalized Hausman-McFadden category-deletion screen immediately before the final fit and stop if its material evidence differs from the approved proposal. Never describe a non-significant screen as proof that IIA holds, and never merge or drop outcome categories automatically.
- For ordinary Poisson, require `count_dispersion_check.model_fitting_allowed: true` before approval and fitting. Recompute the adjusted one-sided overdispersion screen immediately before the final fit and stop if its material evidence differs from the approved proposal. Do not use Pearson dispersion or a fixed rule-of-thumb threshold as an automatic model-selection rule, and never describe a non-significant screen as proof that equidispersion holds.
- For negative-binomial regression, require `negative_binomial_need_check.model_fitting_allowed: true`; recompute it before fitting and stop on material drift. Absence of detected extra dispersion is a reason to return to Poisson, not permission to keep the more flexible model.
- For Poisson and negative-binomial regression, require `zero_inflation_check.model_fitting_allowed: true`; recompute it before fitting and stop on material drift. Detected excess zeros require a dedicated zero-inflated/hurdle workflow and cannot be bypassed by generic approval.
- For prediction goals, require `predictive-validation.status: completed` before any performance claim. Use deterministic five-fold sample-out predictions, stratified folds for classification, and model-appropriate metrics/calibration. Keep the approved predictors, references, and functional forms fixed across folds; forbid outcome-guided tuning. Describe the result as internal cross-validation only, never as external validation or guaranteed new-population performance.
- Include confidence intervals and effect sizes; never decide from p-values alone.
- Describe observational results as associations unless a dedicated causal design has been approved.
- Do not generate `final-report.html` in this skill.
- Do not author, modify, or ask permission to author an extension script. Use only the bundled scripts in this Skill. If they cannot validly analyze the confirmed task, stop and describe the required specialized workflow.
- Do not call external APIs or install packages automatically.
