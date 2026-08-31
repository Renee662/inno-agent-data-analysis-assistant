import { randomUUID } from "node:crypto";

export interface QuestionBridgeAnswer {
	questionIndex: number;
	question: string;
	kind: "option" | "custom" | "chat" | "multi";
	answer: string | null;
	selected?: string[];
	notes?: string;
	preview?: string;
}

export interface QuestionBridgeResult {
	answers: QuestionBridgeAnswer[];
	cancelled: boolean;
	error?: string;
	questionId?: string;
	approvalRecords?: Array<{
		approvalId: string;
		action: string;
		path: string;
		artifactPath: string;
		artifactSha256: string;
	}>;
}

type SseEmitter = (data: unknown) => void;
export type QuestionBridgeLifecycleEvent = "waiting" | "resumed" | "cancelled";
type LifecycleListener = (event: QuestionBridgeLifecycleEvent) => void;

interface PendingQuestion {
	questionId: string;
	resolve: (result: QuestionBridgeResult) => void;
}

class QuestionBridge {
	private emitter: SseEmitter | null = null;
	private pending: PendingQuestion | null = null;
	private lifecycleListeners = new Set<LifecycleListener>();

	subscribeLifecycle(listener: LifecycleListener): () => void {
		this.lifecycleListeners.add(listener);
		return () => this.lifecycleListeners.delete(listener);
	}

	private emitLifecycle(event: QuestionBridgeLifecycleEvent): void {
		for (const listener of this.lifecycleListeners) {
			try {
				listener(event);
			} catch {
				// Timeout bookkeeping must never break questionnaire delivery.
			}
		}
	}

	setEmitter(fn: SseEmitter | null): void {
		this.emitter = fn;
	}

	ask(params: unknown): Promise<QuestionBridgeResult> {
		if (!this.emitter) {
			return Promise.resolve({ answers: [], cancelled: true, error: "no_ui" });
		}

		if (this.pending) {
			this.pending.resolve({ answers: [], cancelled: true, error: "superseded" });
			this.pending = null;
		}

		const questionId = randomUUID();
		const emitter = this.emitter;

		return new Promise<QuestionBridgeResult>((resolve) => {
			this.pending = { questionId, resolve };
			this.emitLifecycle("waiting");
			emitter({ type: "question", questionId, params });
		});
	}

	respond(questionId: string, result: QuestionBridgeResult): boolean {
		if (!this.pending || this.pending.questionId !== questionId) return false;
		const { resolve } = this.pending;
		this.pending = null;
		this.emitLifecycle("resumed");
		resolve({ ...result, questionId });
		return true;
	}

	cancel(): void {
		if (!this.pending) return;
		const { resolve } = this.pending;
		this.pending = null;
		this.emitLifecycle("cancelled");
		resolve({ answers: [], cancelled: true, error: "disconnected" });
	}
}

export const questionBridge = new QuestionBridge();
