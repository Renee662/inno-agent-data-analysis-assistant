type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
	return Boolean(value) && typeof value === "object";
}

function textFromAssistantContent(content: unknown): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";

	return content
		.map((part) => {
			if (!isRecord(part) || part.type !== "text" || typeof part.text !== "string") return "";
			return part.text.trim();
		})
		.filter(Boolean)
		.join("\n")
		.trim();
}

function toolNamesFromAssistantContent(content: unknown): string[] {
	if (!Array.isArray(content)) return [];
	return content.flatMap((part) => {
		if (!isRecord(part) || part.type !== "toolCall") return [];
		return typeof part.name === "string" ? [part.name] : [];
	});
}

/**
 * Return only text that belongs to a terminal assistant response.
 *
 * PI persists an assistant message before every tool call. Models sometimes put
 * planning notes in that message as ordinary text instead of a thinking block.
 * Those messages use stopReason="toolUse". Operational text accompanying
 * inspection/execution calls is hidden. A concise content explanation directly
 * accompanying ask_user_question is retained because it is part of the user's
 * decision context rather than internal execution narration.
 */
export function userFacingTextFromAssistantMessage(message: unknown): string {
	if (!isRecord(message) || message.role !== "assistant") return "";
	if (message.stopReason === "error") return "";
	if (message.stopReason === "toolUse") {
		const toolNames = toolNamesFromAssistantContent(message.content);
		if (toolNames.length !== 1 || toolNames[0] !== "ask_user_question") return "";
	}
	return textFromAssistantContent(message.content);
}
