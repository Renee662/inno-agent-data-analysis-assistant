import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, relative, resolve, sep } from "node:path";
import { checkWorkspaceMutationPath } from "./workspace-path-guard.js";

export type InnoQuestionKind = "decision" | "task-card" | "observation";
export type InnoApprovalAction =
	| "approve-analysis-task"
	| "approve-workflow-support"
	| "approve-model-specification"
	| "approve-final-report";

export interface InnoQuestionOption {
	label: string;
	description: string;
	preview?: string;
	approvalDecision?: "approve" | "revise" | "reject" | "help";
}

export interface InnoQuestionData {
	question: string;
	header: string;
	options: InnoQuestionOption[];
	multiSelect?: boolean;
	questionKind?: InnoQuestionKind;
	approvalAction?: InnoApprovalAction;
	approvalArtifactPath?: string;
	/** Computed by the server when the question is presented; never supplied by the model. */
	approvalArtifactSha256?: string;
	documentPath?: string;
	documentTitle?: string;
	documentCaption?: string;
	imagePath?: string;
	imageAlt?: string;
	imageCaption?: string;
}

export interface InnoQuestionnaireParams {
	questions: InnoQuestionData[];
}

export interface PreparedQuestionnaireResult {
	ok: boolean;
	params?: InnoQuestionnaireParams;
	error?: string;
}

const DOCUMENT_EXTENSIONS = new Set([".json", ".md", ".markdown"]);
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"]);
const MAX_DOCUMENT_BYTES = 2 * 1024 * 1024;

function workspaceRelativeExistingFile(
	workspaceDir: string,
	requestedPath: string,
	extensions: Set<string>,
): { ok: true; path: string; size: number } | { ok: false; error: string } {
	const checked = checkWorkspaceMutationPath(workspaceDir, requestedPath);
	if (!checked.allowed || !checked.resolvedPath || !existsSync(checked.resolvedPath)) {
		return { ok: false, error: `问卷引用的工作区文件不存在或超出工作区：${requestedPath}` };
	}
	const stat = statSync(checked.resolvedPath);
	if (!stat.isFile()) return { ok: false, error: `问卷引用路径不是文件：${requestedPath}` };
	if (!extensions.has(extname(checked.resolvedPath).toLowerCase())) {
		return { ok: false, error: `问卷引用了不支持的文件类型：${requestedPath}` };
	}
	return {
		ok: true,
		path: relative(workspaceDir, checked.resolvedPath).replace(/\\/g, "/"),
		size: stat.size,
	};
}

function sha256File(path: string): string {
	return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/**
 * Enforce presentation contracts before a questionnaire reaches the web UI.
 * Task-card questions always carry the current machine-readable task card;
 * observation questions always carry the plot the user is asked to inspect.
 */
export function prepareQuestionnairePresentation(
	params: InnoQuestionnaireParams,
	workspaceDir: string,
	fallbackTaskCardPath?: string,
): PreparedQuestionnaireResult {
	const questions: InnoQuestionData[] = [];
	for (const original of params.questions) {
		const question = { ...original };
		const questionKind = question.questionKind;
		if (!questionKind) {
			return {
				ok: false,
				error: "数据分析问卷必须显式设置 questionKind；不能再根据问题文字猜测展示类型。",
			};
		}

		if (questionKind === "task-card") {
			const requested = question.documentPath?.trim() || fallbackTaskCardPath?.trim();
			if (!requested) {
				return { ok: false, error: "任务卡确认问卷必须携带 documentPath。" };
			}
			const document = workspaceRelativeExistingFile(workspaceDir, requested, DOCUMENT_EXTENSIONS);
			if (!document.ok) return document;
			if (document.size > MAX_DOCUMENT_BYTES) {
				return { ok: false, error: `任务卡文件过大，无法在问卷中展示：${requested}` };
			}
			question.documentPath = document.path;
			question.documentTitle ||= "分析任务卡";
		}

		if (questionKind === "observation" && !question.imagePath?.trim()) {
			return { ok: false, error: "需要用户观察图形的问卷必须携带 imagePath。" };
		}
		if (question.imagePath?.trim()) {
			const image = workspaceRelativeExistingFile(workspaceDir, question.imagePath, IMAGE_EXTENSIONS);
			if (!image.ok) return image;
			question.imagePath = image.path;
		}

		if (question.approvalAction) {
			const requested = question.approvalArtifactPath?.trim();
			if (!requested) {
				return { ok: false, error: "审批问卷必须携带 approvalArtifactPath。" };
			}
			const artifact = workspaceRelativeExistingFile(workspaceDir, requested, DOCUMENT_EXTENSIONS);
			if (!artifact.ok) return artifact;
			const approveOptions = question.options.filter((option) => option.approvalDecision === "approve");
			if (approveOptions.length !== 1) {
				return { ok: false, error: "审批问卷必须且只能有一个 approvalDecision=approve 的选项。" };
			}
			question.approvalArtifactPath = artifact.path;
			question.approvalArtifactSha256 = sha256File(joinWorkspacePath(workspaceDir, artifact.path));
		}

		questions.push(question);
	}
	return { ok: true, params: { questions } };
}

function joinWorkspacePath(workspaceDir: string, relativePath: string): string {
	return resolve(workspaceDir, relativePath.replace(/\//g, sep));
}
