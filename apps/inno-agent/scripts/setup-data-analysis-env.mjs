import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ENVIRONMENT_ID = "data-analysis-assistant";
const scriptDir = dirname(fileURLToPath(import.meta.url));
const requirementsPath = join(scriptDir, "data-analysis-requirements.txt");
const dataDir = process.env.INNO_DATA_DIR;

if (!dataDir) {
	console.error("INNO_DATA_DIR is not set. Start Inno Agent with its normal runtime arguments first.");
	process.exit(1);
}
if (!existsSync(requirementsPath)) {
	console.error(`Baseline requirements file is missing: ${requirementsPath}`);
	process.exit(1);
}

const environmentDir = join(dataDir, "python-envs", ENVIRONMENT_ID);
const pythonPath =
	process.platform === "win32"
		? join(environmentDir, "Scripts", "python.exe")
		: join(environmentDir, "bin", "python");
const markerPath = join(environmentDir, ".inno-environment.json");
const requirementsHash = createHash("sha256").update(readFileSync(requirementsPath)).digest("hex");

if (process.argv.includes("--status")) {
	let marker = null;
	try {
		marker = JSON.parse(readFileSync(markerPath, "utf8"));
	} catch {
		// Missing or invalid marker means the environment is not ready.
	}
	const ready =
		existsSync(pythonPath) &&
		marker?.environmentId === ENVIRONMENT_ID &&
		marker?.requirementsHash === requirementsHash;
	console.log(
		JSON.stringify(
			{ ready, environmentDir, pythonPath, requirementsPath, requirementsHash, markerPath },
			null,
			2,
		),
	);
	process.exit(ready ? 0 : 2);
}

function run(command, args) {
	const result = spawnSync(command, args, {
		stdio: "inherit",
		windowsHide: true,
		env: {
			...process.env,
			PYTHONUTF8: "1",
			PYTHONIOENCODING: "utf-8",
		},
	});
	if (result.error) throw result.error;
	if (result.status !== 0) {
		throw new Error(`${command} exited with status ${result.status ?? "unknown"}`);
	}
}

function findBasePython() {
	const configured = process.env.INNO_DATA_ANALYSIS_BASE_PYTHON?.trim();
	if (configured) return { command: configured, prefix: [] };

	const candidates =
		process.platform === "win32"
			? [
					{ command: "py", prefix: ["-3"] },
					{ command: "python", prefix: [] },
				]
			: [
					{ command: "python3", prefix: [] },
					{ command: "python", prefix: [] },
				];

	for (const candidate of candidates) {
		const probe = spawnSync(candidate.command, [...candidate.prefix, "--version"], {
			stdio: "ignore",
			windowsHide: true,
		});
		if (!probe.error && probe.status === 0) return candidate;
	}
	throw new Error("Python 3 was not found. Install Python 3 before preparing the data-analysis environment.");
}

try {
	mkdirSync(dirname(environmentDir), { recursive: true });
	if (!existsSync(pythonPath)) {
		const basePython = findBasePython();
		console.log(`Creating shared data-analysis environment: ${environmentDir}`);
		run(basePython.command, [...basePython.prefix, "-m", "venv", environmentDir]);
	}

	console.log("Installing the shared baseline requirements. This is a one-time network operation.");
	run(pythonPath, [
		"-m",
		"pip",
		"install",
		"--disable-pip-version-check",
		"-r",
		requirementsPath,
	]);

	const verification = spawnSync(
		pythonPath,
		[
			"-c",
			"import json,sys,numpy,pandas,scipy,statsmodels,matplotlib,seaborn,openpyxl,xlrd; " +
				"print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__," +
				"'pandas':pandas.__version__,'scipy':scipy.__version__," +
				"'statsmodels':statsmodels.__version__,'matplotlib':matplotlib.__version__," +
				"'seaborn':seaborn.__version__,'openpyxl':openpyxl.__version__," +
				"'xlrd':xlrd.__version__}))",
		],
		{ encoding: "utf8", windowsHide: true },
	);
	if (verification.error) throw verification.error;
	if (verification.status !== 0) {
		throw new Error(verification.stderr || "Baseline import verification failed.");
	}
	const versions = JSON.parse(verification.stdout.trim());

	writeFileSync(
		markerPath,
		`${JSON.stringify(
			{
				environmentId: ENVIRONMENT_ID,
				requirementsHash,
				createdAt: new Date().toISOString(),
				pythonPath,
				versions,
			},
			null,
			2,
		)}\n`,
		"utf8",
	);
	console.log(`DATA_ANALYSIS_ENV_READY ${pythonPath}`);
} catch (error) {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
}
