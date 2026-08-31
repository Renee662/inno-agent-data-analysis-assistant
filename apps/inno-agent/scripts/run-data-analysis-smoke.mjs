#!/usr/bin/env node

import { existsSync, readdirSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "../../..");
const tests = readdirSync(scriptDir)
	.filter((name) => /^smoke-.*\.(?:mjs|py)$/.test(name))
	.sort();

if (process.argv.includes("--list")) {
	for (const test of tests) console.log(test);
	process.exit(0);
}

const managedPythonCandidates = process.platform === "win32"
	? [join(repoRoot, "runtime", "data", "python-envs", "data-analysis-assistant", "Scripts", "python.exe")]
	: [join(repoRoot, "runtime", "data", "python-envs", "data-analysis-assistant", "bin", "python")];
const python = process.env.INNO_DATA_ANALYSIS_PYTHON
	|| managedPythonCandidates.find((candidate) => existsSync(candidate))
	|| process.env.PYTHON
	|| (process.platform === "win32" ? "python" : "python3");

const startedAt = Date.now();
for (const [index, test] of tests.entries()) {
	const testPath = join(scriptDir, test);
	const command = extname(test) === ".py" ? python : process.execPath;
	console.log(`\n[${index + 1}/${tests.length}] ${test}`);
	const result = spawnSync(command, [testPath], {
		cwd: repoRoot,
		stdio: "inherit",
		env: {
			...process.env,
			PYTHONDONTWRITEBYTECODE: "1",
		},
	});
	if (result.error) {
		console.error(`Failed to start ${test}: ${result.error.message}`);
		process.exit(1);
	}
	if (result.status !== 0) {
		console.error(`${test} failed with exit code ${result.status ?? "unknown"}.`);
		process.exit(result.status ?? 1);
	}
}

const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(1);
console.log(`\nAll ${tests.length} data-analysis smoke tests passed in ${elapsedSeconds}s.`);
