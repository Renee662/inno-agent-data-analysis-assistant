import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { RuntimePaths } from "../runtime.js";

export const DATA_ANALYSIS_ENVIRONMENT_ID = "data-analysis-assistant";
export const DATA_ANALYSIS_ENVIRONMENT_APPROVAL = "允许准备数据分析基础环境";

export interface DataAnalysisEnvironmentStatus {
	ready: boolean;
	reason: "ready" | "missing-python" | "missing-marker" | "requirements-changed" | "invalid-marker";
	environmentDir: string;
	pythonPath: string;
	requirementsPath: string;
	requirementsHash: string;
	markerPath: string;
	setupScriptPath: string;
}

interface EnvironmentMarker {
	environmentId?: string;
	requirementsHash?: string;
}

export function resolveDataAnalysisEnvironmentPaths(paths: RuntimePaths) {
	const environmentDir = join(paths.dataDir, "python-envs", DATA_ANALYSIS_ENVIRONMENT_ID);
	const pythonPath =
		process.platform === "win32"
			? join(environmentDir, "Scripts", "python.exe")
			: join(environmentDir, "bin", "python");
	return {
		environmentDir,
		pythonPath,
		requirementsPath: join(paths.codeDir, "scripts", "data-analysis-requirements.txt"),
		markerPath: join(environmentDir, ".inno-environment.json"),
		setupScriptPath: join(paths.codeDir, "scripts", "setup-data-analysis-env.mjs"),
	};
}

function hashRequirements(requirementsPath: string): string {
	if (!existsSync(requirementsPath)) return "";
	return createHash("sha256").update(readFileSync(requirementsPath)).digest("hex");
}

export function inspectDataAnalysisEnvironment(paths: RuntimePaths): DataAnalysisEnvironmentStatus {
	const resolved = resolveDataAnalysisEnvironmentPaths(paths);
	const requirementsHash = hashRequirements(resolved.requirementsPath);
	const base = { ...resolved, requirementsHash };

	if (!existsSync(resolved.pythonPath)) {
		return { ...base, ready: false, reason: "missing-python" };
	}
	if (!existsSync(resolved.markerPath)) {
		return { ...base, ready: false, reason: "missing-marker" };
	}

	try {
		const marker = JSON.parse(readFileSync(resolved.markerPath, "utf8")) as EnvironmentMarker;
		if (
			marker.environmentId !== DATA_ANALYSIS_ENVIRONMENT_ID ||
			typeof marker.requirementsHash !== "string"
		) {
			return { ...base, ready: false, reason: "invalid-marker" };
		}
		if (!requirementsHash || marker.requirementsHash !== requirementsHash) {
			return { ...base, ready: false, reason: "requirements-changed" };
		}
		return { ...base, ready: true, reason: "ready" };
	} catch {
		return { ...base, ready: false, reason: "invalid-marker" };
	}
}

const WORKSPACE_PYTHON_RE =
	/["']?\.?[\\/]\.venv[\\/](?:Scripts[\\/]python(?:\.exe)?|bin[\\/]python)["']?/i;
const WORKSPACE_PYTHON_RE_GLOBAL =
	/["']?\.?[\\/]\.venv[\\/](?:Scripts[\\/]python(?:\.exe)?|bin[\\/]python)["']?/gi;

export function usesWorkspaceLocalPython(command: string): boolean {
	return WORKSPACE_PYTHON_RE.test(command);
}

export function rewriteDataAnalysisPythonCommand(command: string, pythonPath: string): string {
	return command.replace(WORKSPACE_PYTHON_RE_GLOBAL, `"${pythonPath}"`);
}

export function formatDataAnalysisEnvironmentPrompt(status: DataAnalysisEnvironmentStatus): string {
	if (status.ready) {
		return `# 数据分析基础环境

- 基础数据分析环境已经准备完成，并在不同对话和“数据分析助手”工作区之间复用。
- 运行任何 Python Skill 脚本时，必须使用这个解释器：\`${status.pythonPath}\`。
- 不要创建或使用当前工作区的 \`.venv\`，也不要再次安装 pandas、numpy、scipy、statsmodels、matplotlib、seaborn、openpyxl 或 xlrd。
- 如果旧版 Skill 示例仍写着 \`./.venv\` 或 \`.\\.venv\`，以本节给出的共享解释器为准；运行层也会自动改写旧命令。
- Skill 中的依赖清单只用于说明能力和校验版本，不代表每次分析都要安装。`;
	}

	return `# 数据分析基础环境

- 当前共享基础环境尚未准备好（状态：${status.reason}），因此暂时不要运行依赖 pandas 等科学计算包的脚本。
- 这是一次性的环境准备，不是每次分析都需要重复安装。准备完成后，不同对话和“数据分析助手”工作区会共同复用。
- 不要逐个 Skill 询问安装 pandas、openpyxl、xlrd 等包，也不要创建当前工作区的 \`.venv\`。
- 如用户要开始正式分析，请一次性说明需要准备的基础包、网络下载和时间成本，然后要求用户明确回复“${DATA_ANALYSIS_ENVIRONMENT_APPROVAL}”。
- 只有收到上述原句后，才可运行一次 \`node "${status.setupScriptPath}"\`；安装完成后报告环境状态并停止，不要在同一轮继续分析。`;
}
