import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { basename, extname, join, normalize, sep } from "node:path";
import { DATA_ANALYSIS_ENVIRONMENT_APPROVAL } from "./data-analysis-python-env.js";

export const DATA_ANALYSIS_INSTALL_APPROVAL = DATA_ANALYSIS_ENVIRONMENT_APPROVAL;
export const DATA_ANALYSIS_SETUP_TIMEOUT_SECONDS = 900;

export type DataAnalysisScaleTier = "small" | "medium" | "large" | "very-large";
export type DataAnalysisToolBudgetBucket = "inspection" | "execution" | "delivery";

export interface DataAnalysisToolBudgets {
	inspection: number;
	execution: number;
	delivery: number;
}

export interface DataAnalysisRuntimeLimits {
	tier: DataAnalysisScaleTier;
	rows?: number;
	columns?: number;
	estimatedCells?: number;
	sourceBytes?: number;
	commandTimeoutSeconds: number;
	activeBudgetMs: number;
	toolIdleTimeoutMs: number;
	toolBudgets: DataAnalysisToolBudgets;
	reason: "profile" | "source-size" | "default";
}

export interface DataAnalysisConversationPaths {
	sessionId: string;
	folderName: string;
	relativeDir: string;
	absoluteDir: string;
	inputsDir: string;
	workDir: string;
	outputsDir: string;
}

interface ProfileShape {
	profiles?: Array<{
		row_count?: unknown;
		column_count?: unknown;
	}>;
}

const SCALE_LIMITS: Record<
	DataAnalysisScaleTier,
	Pick<DataAnalysisRuntimeLimits, "commandTimeoutSeconds" | "activeBudgetMs" | "toolIdleTimeoutMs" | "toolBudgets">
> = {
	small: {
		commandTimeoutSeconds: 60,
		activeBudgetMs: 10 * 60_000,
		toolIdleTimeoutMs: 90_000,
		toolBudgets: { inspection: 24, execution: 20, delivery: 8 },
	},
	medium: {
		commandTimeoutSeconds: 180,
		activeBudgetMs: 20 * 60_000,
		toolIdleTimeoutMs: 240_000,
		toolBudgets: { inspection: 36, execution: 32, delivery: 10 },
	},
	large: {
		commandTimeoutSeconds: 600,
		activeBudgetMs: 35 * 60_000,
		toolIdleTimeoutMs: 660_000,
		toolBudgets: { inspection: 48, execution: 44, delivery: 12 },
	},
	"very-large": {
		commandTimeoutSeconds: 1_200,
		activeBudgetMs: 60 * 60_000,
		toolIdleTimeoutMs: 1_260_000,
		toolBudgets: { inspection: 60, execution: 56, delivery: 16 },
	},
};

const SCRIPT_FILE_RE = /\.(?:py|r|ipynb|js|mjs|cjs|ts|tsx|sh|bash|zsh|ps1|bat|cmd)$/i;
const INLINE_SCRIPT_RE = /(?:(?:\bpython(?:3|\.exe)?|["'][^"'\r\n]*python(?:3|\.exe)?["']|\$env:INNO_DATA_ANALYSIS_PYTHON)\s+(?:-c\b|-(?=\s|$))|\bnode(?:\.exe)?\s+(?:-e|--eval)\b|\bpowershell(?:\.exe)?\b[^\r\n]*(?:-encodedcommand|-command)\b)/i;
const SCRIPT_WRITE_COMMAND_RE = /(?:>|>>|set-content|add-content|out-file|new-item|copy-item|move-item)[^\r\n]*\.(?:py|r|ipynb|js|mjs|cjs|ts|tsx|sh|bash|zsh|ps1|bat|cmd)\b/i;

function serializedToolInput(input: Record<string, unknown> | undefined): string {
	try {
		return JSON.stringify(input ?? {}).replace(/\\/g, "/").toLowerCase();
	} catch {
		return "";
	}
}

const READ_ONLY_SHELL_COMMANDS = new Set([
	"cat", "convertfrom-json", "dir", "echo", "file", "find", "format-list", "format-table", "get-childitem",
	"get-content", "get-item", "grep", "head", "jq", "ls", "measure-object", "pwd",
	"rg", "select-object", "select-string", "sort-object", "stat", "tail",
	"test-path", "type", "wc", "where", "where-object", "which", "write-output",
]);
const MUTATING_SHELL_RE = /(?:^|[\s|;&])(?:add-content|copy-item|cp|del|erase|git\s+(?:add|commit|push|pull|checkout|switch|reset|clean|merge|rebase)|mkdir|move-item|mv|new-item|out-file|remove-item|ren|rename-item|rm|set-content|tee|touch)(?:\s|$)|(?:^|[^><])>>?|\b(?:npm|pnpm|yarn|pip|pip3|conda|mamba)\s+(?:add|install|remove|uninstall|update)\b/i;

/** True only when every shell pipeline segment is a conservative read-only inspection. */
export function isReadOnlyDataAnalysisShellCommand(command: string): boolean {
	const normalized = command.trim();
	if (!normalized || MUTATING_SHELL_RE.test(normalized)) return false;
	const segments = normalized.split(/\r?\n|&&|\|\||;|\|/).map((part) => part.trim()).filter(Boolean);
	if (segments.length === 0) return false;
	return segments.every((segment) => {
		const lowered = segment
			.replace(/^\s*(?:sudo\s+)?/, "")
			.replace(/^&\s+/, "")
			.toLowerCase();
		if (/^git\s+(?:status|diff|log|show)(?:\s|$)/.test(lowered)) return true;
		const commandName = lowered.match(/^["']?([^\s"']+)/)?.[1]?.replace(/\.exe$/, "");
		return commandName ? READ_ONLY_SHELL_COMMANDS.has(commandName) : false;
	});
}

export function classifyDataAnalysisToolCall(
	toolName: string,
	input?: Record<string, unknown>,
): DataAnalysisToolBudgetBucket | "unmetered" {
	if (toolName === "ask_user_question") return "unmetered";
	const serialized = serializedToolInput(input);
	if (/(?:^|[\/"'])outputs(?:[\/"']|$)/i.test(serialized)) return "delivery";
	if (toolName === "bash") {
		const command = typeof input?.command === "string" ? input.command : "";
		return isReadOnlyDataAnalysisShellCommand(command) ? "inspection" : "execution";
	}
	if (toolName === "write" || toolName === "edit") return "execution";
	return "inspection";
}

export function isDataAnalysisScriptPath(filePath: string): boolean {
	return SCRIPT_FILE_RE.test(filePath.trim());
}

export function attemptsDataAnalysisScriptAuthoring(command: string): boolean {
	return INLINE_SCRIPT_RE.test(command) || SCRIPT_WRITE_COMMAND_RE.test(command);
}

function finiteNonNegativeInteger(value: unknown): number | undefined {
	if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return undefined;
	return Math.floor(value);
}

function classifyScale(estimatedCells?: number, sourceBytes?: number): DataAnalysisScaleTier {
	if (
		(estimatedCells !== undefined && estimatedCells > 20_000_000) ||
		(sourceBytes !== undefined && sourceBytes > 500 * 1024 * 1024)
	) {
		return "very-large";
	}
	if (
		(estimatedCells !== undefined && estimatedCells > 2_000_000) ||
		(sourceBytes !== undefined && sourceBytes > 100 * 1024 * 1024)
	) {
		return "large";
	}
	if (
		(estimatedCells !== undefined && estimatedCells > 100_000) ||
		(sourceBytes !== undefined && sourceBytes > 10 * 1024 * 1024)
	) {
		return "medium";
	}
	return "small";
}

/**
 * Resolve a conservative runtime budget without opening the source dataset.
 * Once profiling exists, row/column counts are authoritative; before that,
 * source-file size provides a cheap fallback.
 */
export function resolveDataAnalysisRuntimeLimits(
	workspaceDir: string,
	artifactRoot = workspaceDir,
): DataAnalysisRuntimeLimits {
	const profilePath = join(artifactRoot, "work", "data-profile", "data-profile.json");
	if (existsSync(profilePath)) {
		try {
			const profile = JSON.parse(readFileSync(profilePath, "utf8")) as ProfileShape;
			const shapes = Array.isArray(profile.profiles) ? profile.profiles : [];
			const rows = shapes.reduce(
				(total, item) => total + (finiteNonNegativeInteger(item.row_count) ?? 0),
				0,
			);
			const columns = shapes.reduce(
				(maximum, item) => Math.max(maximum, finiteNonNegativeInteger(item.column_count) ?? 0),
				0,
			);
			const estimatedCells = shapes.reduce((total, item) => {
				const sheetRows = finiteNonNegativeInteger(item.row_count) ?? 0;
				const sheetColumns = finiteNonNegativeInteger(item.column_count) ?? 0;
				return total + sheetRows * sheetColumns;
			}, 0);
			if (rows > 0 || columns > 0) {
				const tier = classifyScale(estimatedCells);
				return {
					tier,
					rows,
					columns,
					estimatedCells,
					...SCALE_LIMITS[tier],
					reason: "profile",
				};
			}
		} catch {
			// A malformed/incomplete profile falls back to a file-size estimate.
		}
	}

	const taskPath = join(artifactRoot, "work", "analysis-plan", "analysis-task.json");
	let sourceBytes: number | undefined;
	if (existsSync(taskPath)) {
		try {
			const task = JSON.parse(readFileSync(taskPath, "utf8")) as { source_file?: unknown };
			if (typeof task.source_file === "string" && task.source_file.trim()) {
				const directSourcePath = join(workspaceDir, task.source_file);
				const inputSourcePath = join(artifactRoot, "inputs", basename(task.source_file));
				const sourcePath = existsSync(directSourcePath) ? directSourcePath : inputSourcePath;
				if (existsSync(sourcePath)) sourceBytes = statSync(sourcePath).size;
			}
		} catch {
			// Continue with the safe small-data default.
		}
	}
	if (sourceBytes === undefined) {
		try {
			const supported = new Set([".csv", ".tsv", ".xlsx", ".xls"]);
			const inputDir = join(artifactRoot, "inputs");
			const scanDir = existsSync(inputDir) ? inputDir : workspaceDir;
			sourceBytes = readdirSync(scanDir, { withFileTypes: true })
				.filter((entry) => entry.isFile() && supported.has(extname(entry.name).toLowerCase()))
				.reduce(
					(maximum, entry) =>
						Math.max(maximum, statSync(join(scanDir, entry.name)).size),
					0,
				);
			if (sourceBytes === 0) sourceBytes = undefined;
		} catch {
			sourceBytes = undefined;
		}
	}
	const tier = classifyScale(undefined, sourceBytes);
	return {
		tier,
		sourceBytes,
		...SCALE_LIMITS[tier],
		reason: sourceBytes === undefined ? "default" : "source-size",
	};
}

/**
 * Give every conversation in the shared preset workspace its own artifact
 * root. Session filenames already contain a sortable timestamp and UUID; the
 * shorter folder name keeps the file panel readable while remaining unique.
 */
export function resolveDataAnalysisConversationPaths(
	workspaceDir: string,
	rawSessionId: string,
): DataAnalysisConversationPaths {
	const sessionId = basename(rawSessionId || "unsaved-session").replace(/\.jsonl$/i, "");
	const timestampMatch = sessionId.match(
		/^(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})/,
	);
	const uuidMatch = sessionId.match(
		/_([0-9a-f]{8})-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
	);
	const timestamp = timestampMatch
		? `${timestampMatch[1]}-${timestampMatch[2]}-${timestampMatch[3]}_${timestampMatch[4]}-${timestampMatch[5]}-${timestampMatch[6]}`
		: "current";
	const shortId = uuidMatch?.[1]?.toLowerCase() ?? "session";
	const folderName = `${timestamp}_${shortId}`;
	const relativeDir = join("conversations", folderName);
	const absoluteDir = join(workspaceDir, relativeDir);
	return {
		sessionId,
		folderName,
		relativeDir,
		absoluteDir,
		inputsDir: join(absoluteDir, "inputs"),
		workDir: join(absoluteDir, "work"),
		outputsDir: join(absoluteDir, "outputs"),
	};
}

export function formatDataAnalysisConversationPrompt(
	conversation: DataAnalysisConversationPaths,
): string {
	return `# 当前对话的独立产物目录

- 本次对话的所有新生成文件必须集中写入：\`${conversation.relativeDir}\`。
- 用户上传的原始数据和说明文件统一位于：\`${join(conversation.relativeDir, "inputs")}\`，只读使用。
- 中间产物统一写入：\`${join(conversation.relativeDir, "work")}\`。
- 最终报告和交付文件统一写入：\`${join(conversation.relativeDir, "outputs")}\`。
- 只使用本次对话 \`inputs/\` 中的输入，不得读取其他对话的输入文件；同名上传会由系统自动保留为不同版本。
- \`.skills/\` 与 \`agent.md\` 是预设文件，不属于对话产物，不得修改。
- 不得再向根目录的 \`work/\`、\`analysis/\`、\`outputs/\` 或单独的 \`analysis-task.json\` 写入新文件。
- 执行 Skill 示例命令时，用环境变量 \`INNO_CONVERSATION_DIR\` 作为本次产物根目录。`;
}

export function formatDataAnalysisRuntimeLimits(limits: DataAnalysisRuntimeLimits): string {
	const scale =
		limits.reason === "profile"
			? `${limits.rows ?? "?"} 行、约 ${(limits.estimatedCells ?? 0).toLocaleString("zh-CN")} 个单元格`
			: limits.sourceBytes !== undefined
				? `源文件约 ${(limits.sourceBytes / 1024 / 1024).toFixed(1)} MB`
				: "尚无数据画像，采用小数据默认值";
	return `# 当前阶段的数据规模与运行预算

- 判定档位：\`${limits.tier}\`（${scale}）。
- 普通分析命令最长 ${limits.commandTimeoutSeconds} 秒；当前阶段 Agent 有效运行预算约 ${Math.round(limits.activeBudgetMs / 60_000)} 分钟。
- 工具按阶段分别计数：检查/读取 ${limits.toolBudgets.inspection} 次、执行/写入 ${limits.toolBudgets.execution} 次、最终交付专用 ${limits.toolBudgets.delivery} 次；结构化问卷不计数，用户完成确认后下一阶段重新计数。
- 问卷等待时间不计入有效运行预算。若预计仍会超时，应先抽样体检、减少一次性图表数量或拆分阶段，不得静默无限延长。`;
}

export const DATA_ANALYSIS_RUNTIME_PROMPT = `# 数据分析工作区运行边界

- 需要使用工具时直接调用，不得先发送普通正文解释“准备检查什么、为什么读取脚本、如何核对字段或决策”。工具调用前后的工作计划、代码兼容性核对、文件名、内部决策 ID 和调试过程都属于内部信息，不向用户展示。
- 只有当前阶段已经完成、需要用户作决定或必须停止时，才发送普通正文。用户可见说明最多四个短要点：结果、关键依据、影响、下一步；专业实现细节仅在用户主动追问时解释。
- 使用 ask_user_question 时，可以在问卷前保留一段简短的核心说明，说明本阶段发现、关键依据及用户为何需要判断；不得输出脚本分派、文件结构、内部决策 ID、兼容性检查或调试过程。问题和选项仍需包含作出决定所需的具体含义。
- 每次用户确认只执行当前一个阶段；不得把数据体检、清洗、建模和报告生成合并成一次长任务。
- 优先运行 .skills/ 中已经提供并经过约束的脚本；不得在对话目录、work/、analysis/、outputs/ 或根目录编写扩展脚本。不得询问用户是否授权编写脚本，也不得把“授权扩展”作为问卷选项。内置脚本不支持时应停止并说明能力缺口与所需专门工作流。
- 一个阶段最多运行一次主脚本；失败后只允许根据明确错误进行一次小范围修正并重试。仍失败则停止并报告。
- 数据体检、建模和报告生成的最长运行时间由已识别的数据规模动态决定；不得自行覆盖系统给出的本轮上限。
- 不要使用 read 把生成的 PNG 再传给文本模型。依据文件存在性、尺寸、生成日志、绘图数据和统计结果核验图表；PNG 保留给用户在文件面板预览。
- 基础依赖使用 Inno Agent 管理的共享数据分析环境，不得在当前工作区创建 .venv，也不得按 Skill 分别安装依赖。
- 共享环境缺失或版本变化时，列出基础包、用途、预计耗时和网络下载风险，然后停止，并要求用户明确回复“${DATA_ANALYSIS_INSTALL_APPROVAL}”。
- “同意分析方案”“合适”“继续”等一般确认不代表同意安装依赖。
- 获得上述明确回复后，只能运行系统提示中给出的共享环境准备脚本；不得直接运行 pip/conda、安装任意包名、外部 URL、可编辑包或额外软件源。
- 工具调用按检查、执行和最终交付分别计数；ls、rg/grep、只读文件查看等只读 shell 命令计入“检查”，写入或运行分析脚本计入“执行”。ask_user_question 不计入额度，最终交付额度不得被前期探索消耗。
- 数据体检产物生成后必须重新读取行列数并更新数据规模档位；每完成一次用户确认，下一阶段的检查、执行和交付额度重新计数，不与此前阶段共用。
- 达到某类工具、命令或总运行时间上限后，立即汇报已完成文件、未完成步骤和最小下一步，不得伪造结果。`;

const PACKAGE_INSTALL_RE =
	/(?:^|\s)(?:-m\s+pip\s+install|(?:pip3?|uv\s+pip|conda|mamba)(?:\.exe)?\s+install)\b/i;
const VENV_CREATE_RE = /(?:^|\s)-m\s+venv\b/i;
const LEGACY_ROOT_OUTPUT_RE =
	/--output-dir(?:\s+|=)(?:"|')?\.?[\\/](?:work|analysis|outputs)(?:[\\/"'\s]|$)/i;

export function isDataAnalysisWorkspacePath(workspaceDir: string): boolean {
	const normalized = normalize(workspaceDir);
	const parts = normalized.toLowerCase().split(sep);
	return basename(normalized).toLowerCase() === "data-analysis-assistant" && parts.includes(".presets");
}

export function hasDataAnalysisInstallApproval(prompt: string): boolean {
	return prompt.includes(DATA_ANALYSIS_INSTALL_APPROVAL);
}

export function isManagedEnvironmentSetupCommand(command: string, setupScriptPath: string): boolean {
	const trimmed = command.trim().replace(/^&\s*/, "");
	if (/[\r\n]|;|&&|\|\||(?<!\|)\|(?!\|)/.test(trimmed)) return false;

	const match = trimmed.match(/^node(?:\.exe)?\s+(?:"([^"]+)"|'([^']+)'|(\S+))$/i);
	const requestedPath = match?.[1] ?? match?.[2] ?? match?.[3];
	return typeof requestedPath === "string" && normalize(requestedPath) === normalize(setupScriptPath);
}

export function isDependencySetupCommand(command: string, setupScriptPath?: string): boolean {
	return (
		PACKAGE_INSTALL_RE.test(command) ||
		VENV_CREATE_RE.test(command) ||
		(Boolean(setupScriptPath) && isManagedEnvironmentSetupCommand(command, setupScriptPath!))
	);
}

export function writesDataAnalysisLegacyRoot(command: string): boolean {
	return LEGACY_ROOT_OUTPUT_RE.test(command);
}

/**
 * After explicit approval, dependency setup is limited to the app-managed
 * shared-environment setup script. Direct pip/conda commands remain blocked.
 */
export function isAllowlistedDependencySetupCommand(command: string, setupScriptPath: string): boolean {
	return isManagedEnvironmentSetupCommand(command, setupScriptPath);
}
