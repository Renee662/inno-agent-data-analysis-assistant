---
name: generate-final-report
description: Package approved tabular-data profiling, preparation, exploratory analysis, statistical-model, diagnostic, figure, and reproducibility artifacts into a polished offline Chinese final-report.html under the current conversation's outputs directory, plus a hashed deliverable manifest. Use only after run-statistical-analysis has completed and the user has reviewed the modeling results.
---

# Generate Final Report

Create the final local HTML report and copy every reproducibility deliverable into `$env:INNO_CONVERSATION_DIR/outputs/`. Generate `final-report.html` as a standalone file that can be downloaded and opened directly in a browser without the rest of the output folder. Do not refit or alter the model in this stage.

## Preconditions

Require all of the following:

- completed data profile;
- `data_preparation_executed: true` in the preparation log;
- explicitly approved model specification;
- `modeling_executed: true` in the model summary;
- no completed `report-manifest.json` for this output unless the user explicitly approved replacement;
- completed analysis run log whose hashes match the cleaned data, model specification, and preparation log;
- model results, diagnostics, figures, cleaned data, and analysis code.
- categorical-factor omnibus tests and multiplicity-adjusted coefficient results;
- approved category-support screen with every retained factor-level by outcome count;
- deterministic continuous-variable functional-form records, decision provenance, and joint shape tests when OLS, Logistic, Poisson, negative-binomial, multinomial-Logistic, or ordinal-Logistic regression contains continuous predictors;
- a user-reviewed plain-language dataset summary and research question;
- human-readable names and units for the variables used in interpretation.
- a pre-treatment missingness-impact screen, post-preparation selected-field missingness record, and conclusion-scope contract that match the model summary.
- a model-specific influence-diagnostics table whose status distinguishes zero candidates from not evaluated, and whose case-deletion comparisons never overwrite the original fit.
- an IIA check record: it must be `not-applicable` for non-multinomial models, while a multinomial Logistic model must have passed the approved and runtime-recomputed generalized Hausman-McFadden category-deletion screen. Report non-rejection only as “no clear choice-set sensitivity detected in this sample”, never as proof that IIA holds.
- a proportional-odds check record: it must be `not-applicable` for non-ordinal models, while an ordinal model must have passed the approved and runtime-recomputed equal-slopes screen. Report non-rejection only as “no clear violation detected in this sample”, never as proof that the assumption holds.
- a count-dispersion check record: it must be `not-applicable` for non-Poisson models, while an ordinary Poisson model must have passed the approved and runtime-recomputed adjusted overdispersion screen. Keep Pearson dispersion descriptive and report non-rejection as absence of detected evidence, never as proof of equidispersion.
- a negative-binomial-need record: it must support extra dispersion for a negative-binomial model and be `not-applicable` otherwise. Do not present a more flexible count model as justified without this evidence.
- an excess-zero record: Poisson and negative-binomial models must have no detected unexplained excess zeros; other models must be `not-applicable`. A clear screen is not proof that structural zeros are absent.
- a predictive-validation record: prediction goals require completed deterministic five-fold internal sample-out validation; non-prediction goals require `not-applicable`. Never relabel internal cross-validation as external validation.

If an artifact is missing or inconsistent, stop rather than filling gaps with invented content.

## Run

Show the current `model-summary.json` through `ask_user_question` with `questionKind: decision`, `documentPath` and `approvalArtifactPath` pointing to that file, and `approvalAction: approve-final-report`. Mark exactly one option `approvalDecision: approve`; revise/help choices do not authorize generation. Run only with the approval-record path returned by that questionnaire:

```powershell
$run = $env:INNO_CONVERSATION_DIR
& $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\generate-final-report\scripts\generate_report.py" `
  --profile (Join-Path $run "work\data-profile\data-profile.json") `
  --analysis-task (Join-Path $run "work\analysis-plan\approved-analysis-task.json") `
  --preparation-log (Join-Path $run "work\data-preparation-execution\data-preparation-log.json") `
  --missingness-impact (Join-Path $run "work\data-preparation-execution\missingness-impact.csv") `
  --model-spec (Join-Path $run "work\statistical-analysis\approved-model-specification.json") `
  --model-results (Join-Path $run "work\statistical-analysis\results\model-results.csv") `
  --model-diagnostics (Join-Path $run "work\statistical-analysis\results\model-diagnostics.csv") `
  --factor-tests (Join-Path $run "work\statistical-analysis\results\factor-omnibus-tests.csv") `
  --category-support (Join-Path $run "work\statistical-analysis\category-support-screen.csv") `
  --shape-tests (Join-Path $run "work\statistical-analysis\results\continuous-shape-tests.csv") `
  --influence-diagnostics (Join-Path $run "work\statistical-analysis\results\influence-diagnostics.csv") `
  --iia-check (Join-Path $run "work\statistical-analysis\results\iia-check.json") `
  --proportional-odds-check (Join-Path $run "work\statistical-analysis\results\proportional-odds-check.json") `
  --count-dispersion-check (Join-Path $run "work\statistical-analysis\results\count-dispersion-check.json") `
  --negative-binomial-need-check (Join-Path $run "work\statistical-analysis\results\negative-binomial-need-check.json") `
  --zero-inflation-check (Join-Path $run "work\statistical-analysis\results\zero-inflation-check.json") `
  --predictive-validation (Join-Path $run "work\statistical-analysis\results\predictive-validation.json") `
  --model-summary (Join-Path $run "work\statistical-analysis\results\model-summary.json") `
  --analysis-run-log (Join-Path $run "work\statistical-analysis\results\analysis-run-log.json") `
  --figures-dir (Join-Path $run "work\statistical-analysis\results\figures") `
  --cleaned-data (Join-Path $run "work\data-preparation-execution\cleaned-data.csv") `
  --analysis-code ".\.skills\run-statistical-analysis\scripts\run_analysis.py" `
  --approval-record "<path returned by ask_user_question>" `
  --output-dir (Join-Path $run "outputs")
```

Use `--preparation-plan` when the approved preparation-plan JSON should also be included.

## Required report sections

Build `$env:INNO_CONVERSATION_DIR/outputs/final-report.html` with exactly these substantive sections:

1. dataset introduction and research question;
2. variable definitions, units, and uses;
3. data quality and preparation record;
4. exploratory data analysis with descriptive summaries and visualizations;
5. result findings with one real-unit interpretation per modeled term;
6. analysis model: selection rationale, model expression, included factors, uncertainty settings, assumptions, understandable diagnostics, and diagnostic figures whose captions interpret the current run rather than repeat generic textbook definitions;
7. statistical tests: test settings, overall-model test where available, coefficient estimates, confidence intervals, p-values, intuitive conclusions for the current data, significance boundaries, and one or two data-specific multiple-choice questions with collapsed answers;
8. limitations and unsupported conclusions;
9. reproducibility code and generated-file inventory.

Place the complete raw diagnostic record after the generated-file inventory as an appendix. Keep comparison-only or audit-oriented values such as AIC, BIC, parameter count, and residual degrees of freedom out of the main diagnostic summary unless they directly answer the user's question. In the main model-diagnostic area, present each retained indicator as: what question it answers, the current value, and what that value means for this dataset.

Keep long mechanical inventories out of the main narrative. When missing-field lists, preparation actions, deferred items, flagged-row identifiers, or similar details exceed eight items, summarize the count and at most the three most important examples in the relevant main section, then point to the final “技术与诊断明细” appendix. Put complete inventories only in compact bordered detail boxes at the end, using smaller type. Never print hundreds of field names or row identifiers in a figure caption, diagnostic table cell, or ordinary body paragraph.

Interpret effects with direction, magnitude, confidence intervals, and p-values where appropriate. For categorical predictors, report the whole-factor omnibus test before category-level comparisons. Show raw and Benjamini–Hochberg-adjusted p-values, and highlight a category only when both the adjusted whole-factor test and adjusted category comparison pass the approved alpha level. For an OLS, Logistic, Poisson, negative-binomial, multinomial-Logistic, or ordinal-Logistic continuous predictor assigned a quadratic or restricted cubic spline form, present its joint overall and nonlinear-component tests and the model-appropriate adjusted prediction curve with confidence band. For multinomial Logistic, plot every unordered category's adjusted probability; for ordinal Logistic, plot every approved level's adjusted probability. Do not plot, narrate, or test a numeric category-code trend. State the deterministic assignment source and rationale; never imply that the user selected technical terminology or that the form was chosen by the smallest p-value. Do not turn individual basis coefficients into one-unit effects or list them as independent findings; keep basis rows only in the compact audit appendix. For classification models, require `category_support_screen.status: clear` and reproduce every retained category×outcome count only in a compact appendix table. State that separation blocked ordinary fitting before approval and that the small-cell and observations-per-parameter benchmarks were review aids, not automatic scientific truth. Never describe an extreme coefficient as a strong effect when support or separation evidence is unresolved. Never use p-values alone and never describe association as causation.

For classification calibration, show category-aware probability calibration: binary models show the confirmed positive class; multinomial models show one-vs-rest calibration for every unordered category; ordinal models show cumulative calibration at every approved boundary. Use only 0/1 observed event rates on these calibration definitions, not numeric category codes. For prediction goals, use the out-of-fold probabilities and report the cross-validation metrics prominently; for association goals, label the calibration view as in-sample diagnostic evidence rather than predictive performance.

For multinomial Logistic, state the `iia_check` status, the minimum Holm-adjusted deletion p-value, and whether all planned deletions were evaluable. Explain that a clear screen means only that this sample did not reveal marked choice-set sensitivity; it does not prove IIA, and substantively similar or nested alternatives remain a limitation. Keep the complete per-category deletion records in the compact technical appendix rather than repeating the same sentence in the main narrative.

Always distinguish original-data missingness, missingness remaining after preparation, and rows excluded by the model's final complete-case frame. A descriptive missing-versus-observed comparison cannot establish MCAR, MAR, or MNAR. If model estimates were not compared across approved missing-data methods, say so and limit conclusions to the analyzed sample; never claim robustness to multiple imputation or alternative missingness assumptions.

Begin with the dataset context, not statistical roles. The hero must state what the dataset describes. Do not list dependent variables, independent variables, raw column names, or model terminology in the hero.

Present interpretations once in the result-findings section. Use real units such as “学习时间每增加1小时” or “出勤率每提高10个百分点”, include uncertainty, and state the association-not-causation boundary. Do not repeat those interpretations in the coefficient table and do not label content as “通俗”, “专业”, or “30秒读懂”.

Keep the analysis-model and statistical-test sections separate. Explain why the selected model fits the outcome scale and analysis goal before showing its expression, specification, assumptions, diagnostics, and diagnostic figures. Interpret diagnostic values in ordinary Chinese without relying on unexplained abbreviations. Write each diagnostic-figure caption from the current run's values, flagged rows, and uncertainty results; do not use a fixed knowledge-only caption. In the statistical-test section, state the confidence and significance levels, identify the coefficient test, report any available overall-model test, and present coefficient estimates, intervals, and p-values. Add a direct explanation of what the overall and term-level tests imply for this dataset. Explain that significance is not practical importance, certainty, or causality, and note multiple-comparison risk when many terms are tested.

Generate EDA from `cleaned-data.csv` before rendering the report. Include a descriptive summary, distribution plots, outcome-factor relationship plots, and a numeric correlation matrix when applicable. Use the same white, muted blue-gray Chinese academic style and save raster plots at 300 DPI. State that EDA relationships are unadjusted.

## Design requirements

- Use an offline, UTF-8, responsive Chinese page with no CDN or external font dependency.
- Use a white academic-paper layout, restrained blue-gray palette, generous whitespace, clear hierarchy, readable tables, and print-friendly A4 CSS.
- On screen, keep the table of contents sticky on the left for the full report, with a modest outer margin; keep every substantive section to its right. Use a wider desktop canvas instead of allowing later sections to expand across the full viewport.
- Use Microsoft YaHei and Chinese fallback fonts, bold section titles, and comfortably large type.
- Embed every report figure as a Base64 data URL inside `final-report.html`, so downloading the HTML alone preserves all figures.
- Show embedded 300 DPI figures without distortion, but scale their on-page previews to a compact bounded height so one image never dominates the report. A single figure must not expand to the full content width. Give each figure a Chinese caption and provide an in-page click-to-enlarge view without duplicating the image data.
- Keep the original 300 DPI image files in the current conversation's `outputs/figures/` for reuse and auditing.
- Keep relative links to cleaned data, analysis code, result tables, logs, and the manifest. Explain that these non-image links require the accompanying `outputs/` folder.
- Before reporting success, verify that every report `<img>` uses a `data:image/...` source, that its source filename corresponds to an existing file in `outputs/figures/`, and that the page has no external CSS or JavaScript dependency.
- Warn when sensitive-candidate fields remain in the downloadable cleaned data.

## Outputs

Create at least:

```text
$env:INNO_CONVERSATION_DIR/
└── outputs/
    ├── final-report.html
    ├── report-manifest.json
    ├── analysis.py
    ├── cleaned-data.csv
    ├── model-results.csv
    ├── model-diagnostics.csv
    ├── factor-omnibus-tests.csv
    ├── category-support-screen.csv
    ├── continuous-shape-tests.csv
    ├── influence-diagnostics.csv
    ├── iia-check.json
    ├── proportional-odds-check.json
    ├── count-dispersion-check.json
    ├── negative-binomial-need-check.json
    ├── zero-inflation-check.json
    ├── predictive-validation.json
    ├── model-summary.json
    ├── analysis-run-log.json
    ├── approved-analysis-task.json
    ├── report-approval.json
    ├── data-profile.json
    ├── data-preparation-log.json
    ├── missingness-impact.csv
    ├── approved-model-specification.json
    └── figures/
```

Report the output path and stop. Do not start a new analysis or call an external API.

`report-manifest.json` is the single source of truth for report-generation state. `model-summary.json` describes only the completed modeling stage and must not duplicate a `final_report_generated` flag.
