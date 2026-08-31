import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";

export interface ContinuationGuardState {
	sawSuccessfulTerminalAssistant: boolean;
	sawCompletedOverflowCompactionRetry: boolean;
}

export function createContinuationGuardState(): ContinuationGuardState {
	return {
		sawSuccessfulTerminalAssistant: false,
		sawCompletedOverflowCompactionRetry: false,
	};
}

/** Track the exact SDK event sequence that precedes the invalid continue bug. */
export function noteContinuationGuardEvent(
	state: ContinuationGuardState,
	event: AgentSessionEvent,
): void {
	if (event.type === "message_end") {
		const message = event.message as { role?: string; stopReason?: string };
		if (message.role === "assistant") {
			state.sawSuccessfulTerminalAssistant = message.stopReason === "stop";
		}
		return;
	}
	if (
		event.type === "compaction_end" &&
		event.reason === "overflow" &&
		!event.aborted &&
		event.willRetry &&
		event.result
	) {
		state.sawCompletedOverflowCompactionRetry = true;
	}
}

/**
 * PI may classify a successful high-token terminal response as a silent
 * overflow, compact it, and then call continue() even though the last retained
 * message is already a completed assistant answer. The answer and artifacts
 * are valid; only that redundant continuation is invalid.
 */
export function isBenignTerminalContinuationError(
	error: unknown,
	state: ContinuationGuardState,
): boolean {
	const message = error instanceof Error ? error.message : String(error ?? "");
	return (
		message === "Cannot continue from message role: assistant" &&
		state.sawSuccessfulTerminalAssistant &&
		state.sawCompletedOverflowCompactionRetry
	);
}
