#!/usr/bin/env node

import { createServer } from "node:http";
import { logger } from "../dist/logger.js";
import { installFetchLogger } from "../dist/utils/fetch-logger.js";

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

const entries = [];
const originalInfo = logger.info.bind(logger);
const originalWarn = logger.warn.bind(logger);
logger.info = (object, message) => { entries.push({ level: "info", object, message }); };
logger.warn = (object, message) => { entries.push({ level: "warn", object, message }); };

const server = createServer((req, res) => {
	let body = "";
	req.on("data", (chunk) => { body += chunk.toString(); });
	req.on("end", () => {
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(JSON.stringify({ ok: true, token: "response-secret", received: body.length }));
	});
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const address = server.address();
const port = typeof address === "object" && address ? address.port : 0;
installFetchLogger({ getSessionId: () => "session-privacy-test" });

try {
	delete process.env.INNO_DEBUG_LOG_BODIES;
	await fetch(`http://127.0.0.1:${port}/chat/completions`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ model: "test-model", apiKey: "request-secret", messages: [{ content: "private body" }] }),
	});
	await new Promise((resolve) => setTimeout(resolve, 50));
	const defaultEntries = entries.splice(0);
	assert(defaultEntries.length === 2, "Default request should emit start and finish metadata");
	assert(defaultEntries.every((entry) => !("requestBody" in entry.object) && !("responseBody" in entry.object)), "Default logs must not contain bodies");
	const finished = defaultEntries.find((entry) => entry.object.event === "llm_request_finished");
	assert(finished?.object.success === true && finished.object.outputChars > 0, "Default finish metadata is incomplete");
	assert(finished?.object.sessionId === "session-privacy-test" && finished.object.model === "test-model", "Session/model metadata is missing");

	process.env.INNO_DEBUG_LOG_BODIES = "true";
	await fetch(`http://127.0.0.1:${port}/messages`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ model: "debug-model", token: "request-secret", prompt: "debug body" }),
	});
	await new Promise((resolve) => setTimeout(resolve, 50));
	const debugText = JSON.stringify(entries);
	assert(debugText.includes("[REDACTED]"), "Debug bodies must redact sensitive fields");
	assert(!debugText.includes("request-secret") && !debugText.includes("response-secret"), "Sensitive values leaked in debug logs");
	console.log(JSON.stringify({ ok: true, checks: 6 }));
} finally {
	logger.info = originalInfo;
	logger.warn = originalWarn;
	server.close();
	delete process.env.INNO_DEBUG_LOG_BODIES;
}
