/**
 * Keep image blocks in the persisted session/UI, but remove them from the
 * transient context sent to a text-only model.
 *
 * PI exposes a deep copy of the context to extension handlers, so returning
 * these rewritten messages does not delete generated images or alter the
 * conversation history shown in the web UI.
 */

export const TEXT_ONLY_IMAGE_NOTICE =
	"[图片内容未发送给当前文本模型。图片文件仍保存在工作区，可在右侧文件面板预览。请依据生成代码、表格和数值结果继续分析。]";

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isImageBlock(value: unknown): boolean {
	return isRecord(value) && value.type === "image";
}

export interface TextOnlyMessageSanitizeResult<T> {
	messages: T[];
	removedImageCount: number;
}

/**
 * Remove PI `image` content blocks from messages sent to text-only models.
 * A short text notice is appended to every affected message so image-only
 * tool results do not become empty or lose their meaning.
 */
export function sanitizeMessagesForTextOnly<T>(
	messages: readonly T[],
): TextOnlyMessageSanitizeResult<T> {
	let removedImageCount = 0;
	const sanitized = messages.map((message) => {
		if (!isRecord(message) || !Array.isArray(message.content)) return message;

		const imageCount = message.content.filter(isImageBlock).length;
		if (imageCount === 0) return message;

		removedImageCount += imageCount;
		const content = message.content.filter((block) => !isImageBlock(block));
		const alreadyHasNotice = content.some(
			(block) => isRecord(block) && block.type === "text" && block.text === TEXT_ONLY_IMAGE_NOTICE,
		);
		if (!alreadyHasNotice) {
			content.push({ type: "text", text: TEXT_ONLY_IMAGE_NOTICE });
		}

		return {
			...message,
			content,
		} as T;
	});

	return {
		messages: sanitized,
		removedImageCount,
	};
}
