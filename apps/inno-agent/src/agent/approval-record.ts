import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import type { InnoQuestionData, InnoQuestionnaireParams } from "./question-presentation.js";
import type { QuestionBridgeAnswer, QuestionBridgeResult } from "./question-bridge.js";
import { checkWorkspaceMutationPath } from "./workspace-path-guard.js";

export interface ApprovalRecordReference {
	approvalId: string;
	action: string;
	path: string;
	artifactPath: string;
	artifactSha256: string;
}

export interface ApprovalRecordContext {
	workspaceDir: string;
	sessionId: string;
	questionId: string;
	source: "web-question-dialog" | "tui-questionnaire";
}

function sha256File(path: string): string {
	return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function selectedLabels(answer: QuestionBridgeAnswer): string[] {
	if (answer.kind === "multi") return answer.selected ?? [];
	return typeof answer.answer === "string" ? [answer.answer] : [];
}

function approvalSelected(question: InnoQuestionData, answer: QuestionBridgeAnswer): boolean {
	const labels = new Set(selectedLabels(answer));
	return question.options.some(
		(option) => option.approvalDecision === "approve" && labels.has(option.label),
	);
}

function safeSessionSegment(sessionId: string): string {
	return sessionId.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 160) || "local-session";
}

/**
 * Persist server-issued approval receipts only after a structured questionnaire
 * response selected the one option marked as approval. The receipt binds the
 * UI event to the exact bytes shown to the user.
 */
export function recordQuestionApprovals(
	params: InnoQuestionnaireParams,
	result: QuestionBridgeResult,
	context: ApprovalRecordContext,
): QuestionBridgeResult {
	if (result.cancelled) return result;
	const references: ApprovalRecordReference[] = [];

	for (const answer of result.answers) {
		const question = params.questions[answer.questionIndex];
		if (!question?.approvalAction || !approvalSelected(question, answer)) continue;
		const artifactPath = question.approvalArtifactPath;
		const presentedHash = question.approvalArtifactSha256;
		if (!artifactPath || !presentedHash) {
			return { ...result, cancelled: true, error: "approval_contract_missing" };
		}

		const checked = checkWorkspaceMutationPath(context.workspaceDir, artifactPath);
		if (!checked.allowed || !checked.resolvedPath || !existsSync(checked.resolvedPath)) {
			return { ...result, cancelled: true, error: "approval_artifact_missing" };
		}
		if (!statSync(checked.resolvedPath).isFile()) {
			return { ...result, cancelled: true, error: "approval_artifact_invalid" };
		}
		const currentHash = sha256File(checked.resolvedPath);
		if (currentHash !== presentedHash) {
			return { ...result, cancelled: true, error: "approval_artifact_changed" };
		}

		const approvalId = randomUUID();
		const approvalDir = join(
			context.workspaceDir,
			".approvals",
			safeSessionSegment(context.sessionId),
		);
		mkdirSync(approvalDir, { recursive: true });
		const recordPath = join(approvalDir, `${question.approvalAction}-${approvalId}.json`);
		const record = {
			schemaVersion: 1,
			approvalId,
			status: "approved",
			action: question.approvalAction,
			source: context.source,
			questionId: context.questionId,
			sessionId: context.sessionId,
			approvedAt: new Date().toISOString(),
			artifact: {
				path: artifactPath,
				sha256: currentHash,
			},
			decision: {
				questionIndex: answer.questionIndex,
				header: question.header,
				question: question.question,
				selectedLabels: selectedLabels(answer),
				notes: answer.notes,
			},
		};
		writeFileSync(recordPath, `${JSON.stringify(record, null, 2)}\n`, { encoding: "utf-8", flag: "wx" });
		references.push({
			approvalId,
			action: question.approvalAction,
			path: relative(context.workspaceDir, recordPath).replace(/\\/g, "/"),
			artifactPath,
			artifactSha256: currentHash,
		});
	}

	return references.length > 0 ? { ...result, approvalRecords: references } : result;
}
