---
name: tabular-data-profiler
description: Inspect uploaded CSV, TSV, XLSX, and XLS tables plus local companion documentation before statistical analysis. Discover README files, codebooks, questionnaires, metadata, and data-description documents near the table; preserve source attribution, produce a structural profile and local dataset-context inventory, then stop for user confirmation. Do not fit models or modify source data.
---

# Profile Tabular Data

## Workflow

1. Locate the user-provided primary table only inside `$env:INNO_CONVERSATION_DIR/inputs/`. Accept only `.csv`, `.tsv`, `.xlsx`, or `.xls`. Treat that conversation-specific input directory as the default context-search boundary; never search another conversation's inputs.
2. Treat the primary table and every companion file as read-only. Never rename, move, overwrite, clean, merge, or convert them during profiling.
3. If Python dependencies are missing, stop and report the missing packages. Use only the Inno Agent managed shared-environment setup flow from the system prompt after explicit user approval; never install from this Skill's `requirements.txt` directly.
4. Run the bundled profiler from the workspace root with the managed shared environment. Put outputs under this conversation's artifact directory:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\tabular-data-profiler\scripts\profile_data.py" "<relative-table-path>" --output-dir (Join-Path $run "work\data-profile")
   ```

   Use `--sheet "<sheet-name>"` only when the user selects one Excel sheet. Without it, profile every sheet.
5. Read `table-read-spec.json` before interpreting types or columns. The profiler first reads a bounded raw preview with no assumed header and compares first-row, later-row, multi-row, and headerless candidates. For each sheet:
   - continue automatically only when `status` is `auto-confirmed` and `requires_user_confirmation` is `false`;
   - when confirmation is required, show the recommended header row(s), data-start row, and a small column-name preview through `ask_user_question`;
   - offer “采用推荐结构”, “保留第一行作为表头”, “我来指定表头位置”, and “我不理解表头与数据起始行的区别”; do not ask the user to explain every column;
   - a help or custom response is not confirmation.

   After explicit confirmation, re-run the profiler for that sheet. For example, when Excel row 2 is the real header and data begins on row 3:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\tabular-data-profiler\scripts\profile_data.py" "<relative-table-path>" --sheet "Data" --header-rows "2" --structure-confirmed --output-dir (Join-Path $run "work\data-profile")
   ```

   Use `--header-rows "1,2"` for a confirmed two-row header, or `--headerless --structure-confirmed` for a confirmed headerless table. Do not advance while the selected analysis sheet remains `pending-user-confirmation`.
6. Run the bundled local-context discovery script only after the selected table structure is confirmed. Point `--context-root` to the primary table's containing upload directory, not the repository root:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\tabular-data-profiler\scripts\discover_context.py" `
     --table "<relative-table-path>" `
     --profile (Join-Path $run "work\data-profile\data-profile.json") `
     --context-root "<relative-upload-directory>" `
     --output-dir (Join-Path $run "work\data-profile")
   ```

   The script inventories nearby README, Markdown, TXT, JSON/YAML, DOCX, PDF, PPTX, image, and secondary tabular files. It extracts bounded local text from plain-text files and DOCX without network access. It also carries forward codebook-like workbook sheets detected by the profiler.
7. Read `data-profile.md`, `data-profile.json`, `data-dictionary.csv`, `table-read-spec.json`, `dataset-context.md`, and `dataset-context.json`.
8. For a context record marked `requires_parse_document`, use the local `parse_document` tool when its filename indicates a README, codebook, questionnaire, metadata, source note, or when the user identifies it as relevant. Parse the highest-relevance files first and no more than five files in one pass; list the remainder instead of flooding the model context. Do not parse an unrelated image or secondary table merely because it shares the directory.
9. Create `$env:INNO_CONVERSATION_DIR/work/data-profile/dataset-context-evidence.md`. For each used source, record:
   - relative source path;
   - the dataset name, source, version, unit of analysis, field meaning, code, or unit explicitly stated by that source;
   - exact table columns that match the source;
   - contradictions, missing columns, version uncertainty, and unresolved meanings;
   - whether the statement is source-declared or only an Agent hypothesis.
10. If the user explicitly supplied the name of a public dataset and local documentation is insufficient, run the metadata-only verifier automatically; do not show a structured network-permission question. Before the tool call, state in one short progress update that only the public dataset name will be sent and that the uploaded table, rows, column names, paths, and research question remain local. If the user explicitly asked not to use the network, skip lookup and leave identity unresolved.
11. Quote the user-provided name:

   ```powershell
   $run = $env:INNO_CONVERSATION_DIR
   & $env:INNO_DATA_ANALYSIS_PYTHON ".\.skills\tabular-data-profiler\scripts\verify_public_dataset.py" `
     --profile (Join-Path $run "work\data-profile\data-profile.json") `
     --dataset-name "<public-dataset-name>" `
     --output-dir (Join-Path $run "work\data-profile")
   ```

   Add `--claimed-version "<version-or-year>"` only when the user supplied it, and `--sheet` when the profile contains multiple data sheets. The script searches OpenML public metadata and feature definitions, never downloads the public raw dataset, and never sends uploaded data or uploaded column names. If OpenML is unavailable or returns no candidate, report that limitation and ask for a source URL, repository ID, or uploaded codebook; do not treat lookup failure as evidence about identity.
12. For a public source outside OpenML, read only a user-approved official/repository page using an available safe retrieval tool. Record its title, publisher, URL, version, dataset ID, and complete published field list in `$env:INNO_CONVERSATION_DIR/work/data-profile/public-source-record.json`, then run the same verifier with `--source-record`. If no safe retrieval tool is available, ask the user to upload the codebook rather than using an unverified recollection. The record schema is:

   ```json
   {
     "dataset_name": "source-declared name",
     "dataset_id": "stable repository ID",
     "source_title": "page or codebook title",
     "publisher": "repository or publisher",
     "source_url": "https://...",
     "version": "version, release, or year; use unknown when absent",
     "columns": ["exact", "published", "column", "names"]
   }
   ```
13. Read `public-dataset-verification.json` and present at most three candidates. For each candidate, cite its source URL and state the repository ID, source version, lookup time, matched fields, fields missing from the upload, and extra uploaded fields. Interpret statuses conservatively:
   - `strong-candidate`: name and columns are consistent, but identity still requires version/user confirmation;
   - `partial-column-match`: possible subset, derived file, or different release; keep identity unresolved;
   - `insufficient-column-match`: do not use that codebook for field meanings.

   `identity_confirmed` must remain `false` until the user confirms the source/version and all analysis-relevant column conflicts are resolved. A familiar name, filename, or partial overlap is never enough.
14. Summarize the structural profile without exposing full sensitive records. Report file structure, sheet names, row and column counts, inferred types, missingness, duplicates, constants, near-constants, mixed-type columns, possible outliers, and sensitive-field candidates.
15. Use companion documentation and public sources as evidence, not proof. Present candidate meanings for the dependent variable, independent variables, identifiers, categorical variables, and possible controls only as hypotheses requiring user confirmation unless a source explicitly defines them and its version and columns match.
16. End with the required stage summary:
   - 已完成
   - 主要发现
   - 建议
   - 待确认
17. Use `ask_user_question` to confirm the dataset description, public source/version when applicable, unit of analysis, and uncertain variable meanings. Include a concept-specific uncertainty option such as “我不确定每行代表什么”; do not request a numbered or typed confirmation in ordinary chat.
18. Stop after profiling. Do not choose a final model, clean data, or begin statistical analysis until the user confirms variable meanings and the next step.

## Guardrails

- Keep all generated profile artifacts under `$env:INNO_CONVERSATION_DIR/work/data-profile/`.
- Treat `$env:INNO_CONVERSATION_DIR/inputs/` as the only valid upload root for the current analysis. Same-named files may have system-added version suffixes; use the exact returned path.
- Search only the primary table's upload directory, with the discovery script's bounded depth and file-count limits. Do not scan the repository, runtime data, other conversations, or unrelated workspaces for background documents.
- Do not copy raw rows into chat or reports. Show at most masked examples already produced by the profiler.
- Do not treat a same-looking filename, a shared abbreviation, or partial column overlap as dataset identification.
- Do not automatically join, concatenate, or replace the primary table with a discovered secondary table.
- Keep every extracted claim linked to its local source path. If two sources disagree, report the conflict and leave the meaning unresolved.
- Treat name-based sensitive detection as a warning, not proof.
- State when a type or semantic role is uncertain.
- Do not infer causality from column names or correlations.
- Except for the automatic public-metadata lookup triggered by an explicitly supplied public dataset name, do not call external APIs or upload data outside the workspace. This lookup does not require a separate permission question and may send only the dataset name and public catalog IDs; never send the uploaded table, its rows, its column names, local paths, or analysis question.
- If the script fails, report the exact error and propose a fix; never invent profile results.

## Output Contract

Expect these files:

- `data-profile.md`: human-readable data-quality overview.
- `data-profile.json`: machine-readable workbook and column profile.
- `data-dictionary.csv`: one row per column with inferred type and quality indicators.
- `table-read-spec.json`: versioned, source-hash-bound sheet/header/data-start specification with scored candidates and confirmation status.
- `dataset-context.json`: bounded local inventory, extracted excerpts, column mentions, codebook-like workbook sheets, and files that still require `parse_document`.
- `dataset-context.md`: human-readable context-file inventory.
- `dataset-context-evidence.md`: source-attributed facts and unresolved conflicts recorded after the Agent reads relevant documents.
- `public-dataset-verification.json` / `.md`: optional source-attributed public-catalog candidates and deterministic column/version comparison; neither file confirms identity by itself.

Use these outputs as evidence for the next conversation stage. They are intermediate artifacts and do not replace the final `$env:INNO_CONVERSATION_DIR/outputs/final-report.html` required after a complete analysis.
