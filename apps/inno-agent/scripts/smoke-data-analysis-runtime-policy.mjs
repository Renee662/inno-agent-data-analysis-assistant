#!/usr/bin/env node

import {
  attemptsDataAnalysisScriptAuthoring,
  classifyDataAnalysisToolCall,
  isDataAnalysisScriptPath,
  isReadOnlyDataAnalysisShellCommand,
  resolveDataAnalysisRuntimeLimits,
} from "../dist/agent/data-analysis-runtime-policy.js";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(classifyDataAnalysisToolCall("ask_user_question", {}) === "unmetered", "Questions must not consume a budget");
assert(classifyDataAnalysisToolCall("read", { path: "conversations/x/inputs/a.csv" }) === "inspection", "Input reads must use the inspection budget");
assert(classifyDataAnalysisToolCall("bash", { command: "ls -la | grep csv" }) === "inspection", "Read-only shell inspection must use the inspection budget");
assert(classifyDataAnalysisToolCall("bash", { command: "ls -la && echo done && rg -n profile work" }) === "inspection", "Read-only command chains must use the inspection budget");
assert(classifyDataAnalysisToolCall("bash", { command: "rg -n profile work" }) === "inspection", "rg must use the inspection budget");
assert(classifyDataAnalysisToolCall("bash", { command: "python script.py" }) === "execution", "Analysis commands must use the execution budget");
assert(classifyDataAnalysisToolCall("bash", { command: "tool --output-dir conversations/x/outputs" }) === "delivery", "Output commands must use the delivery reserve");
assert(isReadOnlyDataAnalysisShellCommand("Get-Content file.json | Select-String rows"), "PowerShell read-only pipelines must be recognized");
assert(!isReadOnlyDataAnalysisShellCommand("Get-Content file.json | Set-Content changed.json"), "Mutating pipelines must not be treated as inspection");
assert(isDataAnalysisScriptPath("work/custom.py"), "Python authoring must be recognized");
assert(!isDataAnalysisScriptPath("work/result.json"), "Data artifacts must remain writable");
assert(attemptsDataAnalysisScriptAuthoring("python -c print(1)"), "Inline Python must be blocked");
assert(!attemptsDataAnalysisScriptAuthoring("python -m pip --version"), "Normal Python module flags must not be mistaken for inline code");
assert(attemptsDataAnalysisScriptAuthoring("Set-Content work/custom.py"), "Shell script authoring must be blocked");
assert(
  !attemptsDataAnalysisScriptAuthoring(
    '& $env:INNO_DATA_ANALYSIS_PYTHON ".\\.skills\\run-statistical-analysis\\scripts\\run_analysis.py"',
  ),
  "Bundled Skill scripts must remain executable",
);

const temp = mkdtempSync(join(tmpdir(), "inno-runtime-policy-"));
try {
  const work = join(temp, "work", "data-profile");
  mkdirSync(work, { recursive: true });
  writeFileSync(
    join(work, "data-profile.json"),
    JSON.stringify({ profiles: [{ row_count: 32537, column_count: 15 }] }),
  );
  const medium = resolveDataAnalysisRuntimeLimits(temp, temp);
  assert(medium.tier === "medium", "Adult-sized example data must use the medium tier");
  assert(medium.toolBudgets.execution === 32, "Medium execution budget must cover iterative modeling gates");
  assert(medium.activeBudgetMs === 20 * 60_000, "Medium active budget must be 20 minutes");
} finally {
  rmSync(temp, { recursive: true, force: true });
}

console.log(JSON.stringify({ ok: true, checks: 18 }));
