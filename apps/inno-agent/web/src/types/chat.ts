export interface ChatMessage {
	role: "user" | "assistant";
	content: string;
	timestamp: number;
	thinking?: string;
	tools?: ChatToolRecord[];
	channel?: string;
	images?: Array<{ previewUrl: string; mimeType: string }>;
	/** Backend/model error surfaced for this turn (e.g. HTTP 413 over-long context). */
	error?: string;
}

export interface ChatToolRecord {
	toolCallId: string;
	toolName: string;
	args: unknown;
	result?: unknown;
	isError?: boolean;
}

export interface WorkspaceFileChange {
	path: string;
	change: "created" | "modified" | "deleted";
}

// --- Question types ---

export interface QuestionOption {
	label: string;
	description: string;
	preview?: string;
	approvalDecision?: "approve" | "revise" | "reject" | "help";
}

export interface QuestionData {
	question: string;
	header: string;
	options: QuestionOption[];
	multiSelect?: boolean;
	questionKind?: "decision" | "task-card" | "observation";
	approvalAction?: "approve-analysis-task" | "approve-workflow-support" | "approve-model-specification" | "approve-final-report";
	approvalArtifactPath?: string;
	approvalArtifactSha256?: string;
	/** Workspace-relative JSON/Markdown shown as a compact questionnaire card. */
	documentPath?: string;
	documentTitle?: string;
	documentCaption?: string;
	/** Workspace-relative image displayed with this question. */
	imagePath?: string;
	imageAlt?: string;
	imageCaption?: string;
}

export interface PendingQuestion {
	questionId: string;
	params: { questions: QuestionData[] };
}

export interface QuestionAnswer {
	questionIndex: number;
	question: string;
	kind: "option" | "custom" | "chat" | "multi";
	answer: string | null;
	selected?: string[];
	notes?: string;
	preview?: string;
}

export interface QuestionnaireResult {
	answers: QuestionAnswer[];
	cancelled: boolean;
	error?: string;
}

// Phase 2 SSE event types
export type ChatStreamEvent =
	| { type: "text_delta"; delta: string }
	| { type: "thinking_delta"; delta: string }
	| { type: "tool_call_delta"; toolCallId: string; toolName: string; args?: unknown; argsDelta?: string }
	| { type: "tool_start"; toolCallId: string; toolName: string; args: unknown }
	| { type: "tool_end"; toolCallId: string; toolName: string; result: unknown; isError: boolean }
	| { type: "workspace_change"; changes: WorkspaceFileChange[]; toolCallId?: string; toolName?: string; workspaceId?: string; truncated?: boolean }
	| { type: "question"; questionId: string; params: { questions: QuestionData[] } }
	| { type: "done"; fullText: string }
	| { type: "error"; message: string };
