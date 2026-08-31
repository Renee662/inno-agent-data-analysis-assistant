/** Privacy-preserving fetch telemetry for LLM provider calls. */
import { logger } from "../logger.js";

const LLM_API_PATTERNS = ["/chat/completions", "/messages", "/api/stream"];
const MAX_DEBUG_BODY_LENGTH = 8000;
const MAX_DEBUG_RESPONSE_LENGTH = 4000;
const SENSITIVE_KEY_RE = /(?:api[-_]?key|authorization|token|secret|password|cookie|credential)/i;

type FetchFn = typeof globalThis.fetch;

export interface FetchLoggerOptions {
	getSessionId?: () => string;
}

let nextSeq = 1;

function nextReqId(): string {
	return `${nextSeq++}/${Math.floor(Date.now() / 1000)}`;
}

function debugBodiesEnabled(): boolean {
	return /^(?:1|true|yes|on)$/i.test(process.env.INNO_DEBUG_LOG_BODIES ?? "");
}

function safeSessionId(options: FetchLoggerOptions): string {
	try {
		return options.getSessionId?.() ?? "";
	} catch {
		return "";
	}
}

function resolveURL(input: Parameters<FetchFn>[0]): string {
	if (typeof input === "string") return input;
	if (input instanceof URL) return input.href;
	if (input != null && typeof input === "object" && "url" in input) {
		return String((input as { url: unknown }).url);
	}
	return String(input);
}

function extractBodyString(body: NonNullable<Parameters<FetchFn>[1]>["body"]): string {
	if (body == null) return "";
	if (typeof body === "string") return body;
	if (body instanceof Uint8Array || body instanceof ArrayBuffer) return new TextDecoder().decode(body);
	return "[non-text body]";
}

function redactValue(value: unknown, key = ""): unknown {
	if (SENSITIVE_KEY_RE.test(key)) return "[REDACTED]";
	if (Array.isArray(value)) return value.map((item) => redactValue(item));
	if (value && typeof value === "object") {
		return Object.fromEntries(
			Object.entries(value as Record<string, unknown>).map(([childKey, childValue]) => [
				childKey,
				redactValue(childValue, childKey),
			]),
		);
	}
	return value;
}

function redactedDebugBody(body: string, limit: number): string {
	let value = body;
	try {
		value = JSON.stringify(redactValue(JSON.parse(body)));
	} catch {
		value = body.replace(
			/(api[-_]?key|authorization|token|secret|password|cookie)\s*[:=]\s*([^\s,;]+)/gi,
			"$1=[REDACTED]",
		);
	}
	return value.length > limit ? `${value.slice(0, limit)}...[truncated]` : value;
}

function modelFromBody(body: string): string {
	try {
		const parsed = JSON.parse(body) as Record<string, unknown>;
		return typeof parsed.model === "string" ? parsed.model : "";
	} catch {
		return "";
	}
}

function errorType(err: unknown): string {
	if (err instanceof Error) {
		const code = "code" in err ? String(err.code) : "";
		return code ? `${err.name}:${code}` : err.name;
	}
	return typeof err;
}

async function readResponseTelemetry(response: Response, debug: boolean): Promise<{ outputChars: number; responseBody?: string }> {
	const clone = response.clone();
	if (!clone.body) return { outputChars: 0, ...(debug ? { responseBody: "" } : {}) };
	const reader = clone.body.getReader();
	const decoder = new TextDecoder();
	let outputChars = 0;
	let captured = "";
	try {
		while (true) {
			const chunk = await reader.read();
			if (chunk.done) break;
			const text = decoder.decode(chunk.value, { stream: true });
			outputChars += text.length;
			if (debug && captured.length < MAX_DEBUG_RESPONSE_LENGTH) {
				captured += text.slice(0, MAX_DEBUG_RESPONSE_LENGTH - captured.length);
			}
		}
		const tail = decoder.decode();
		outputChars += tail.length;
		if (debug && captured.length < MAX_DEBUG_RESPONSE_LENGTH) captured += tail;
	} finally {
		reader.releaseLock();
	}
	return {
		outputChars,
		...(debug ? { responseBody: redactedDebugBody(captured, MAX_DEBUG_RESPONSE_LENGTH) } : {}),
	};
}

/**
 * Default logs contain metadata only. Set INNO_DEBUG_LOG_BODIES=true to add
 * truncated, field-redacted request and response bodies for local debugging.
 */
export function installFetchLogger(options: FetchLoggerOptions = {}): void {
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async function (input, init): ReturnType<FetchFn> {
		const url = resolveURL(input);
		const method = (init?.method ?? "GET").toString().toUpperCase();
		const isLlmCall = method === "POST" && LLM_API_PATTERNS.some((pattern) => url.includes(pattern));
		if (!isLlmCall) return originalFetch.call(globalThis, input, init) as ReturnType<FetchFn>;

		const requestId = nextReqId();
		const startedAt = new Date();
		const body = extractBodyString(init?.body);
		const debug = debugBodiesEnabled();
		const base = {
			requestId,
			sessionId: safeSessionId(options) || undefined,
			model: modelFromBody(body) || undefined,
			startedAt: startedAt.toISOString(),
			inputChars: body === "[non-text body]" ? undefined : body.length,
		};
		logger.info({
			...base,
			event: "llm_request_started",
			...(debug ? { requestBody: redactedDebugBody(body, MAX_DEBUG_BODY_LENGTH) } : {}),
		}, "LLM request started");

		try {
			const response = await originalFetch.call(globalThis, input, init) as Response;
			const elapsedMs = Date.now() - startedAt.getTime();
			void readResponseTelemetry(response, debug)
				.then((telemetry) => {
					logger[response.ok ? "info" : "warn"]({
						...base,
						event: "llm_request_finished",
						elapsedMs,
						success: response.ok,
						status: response.status,
						errorType: response.ok ? undefined : `HTTP_${response.status}`,
						...telemetry,
					}, "LLM request finished");
				})
				.catch(() => {
					logger[response.ok ? "info" : "warn"]({
						...base,
						event: "llm_request_finished",
						elapsedMs,
						success: response.ok,
						status: response.status,
						errorType: response.ok ? undefined : `HTTP_${response.status}`,
					}, "LLM request finished");
				});
			return response;
		} catch (err) {
			logger.warn({
				...base,
				event: "llm_request_finished",
				elapsedMs: Date.now() - startedAt.getTime(),
				success: false,
				errorType: errorType(err),
				...(debug && err instanceof Error
					? { errorDetails: redactedDebugBody(err.message, MAX_DEBUG_RESPONSE_LENGTH) }
					: {}),
			}, "LLM request failed");
			throw err;
		}
	} as FetchFn;
}
