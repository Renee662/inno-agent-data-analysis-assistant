#!/usr/bin/env node

import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer as createNetServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

function freePort() {
	return new Promise((resolve, reject) => {
		const server = createNetServer();
		server.once("error", reject);
		server.listen(0, "127.0.0.1", () => {
			const address = server.address();
			const port = typeof address === "object" && address ? address.port : 0;
			server.close((error) => error ? reject(error) : resolve(port));
		});
	});
}

async function waitForHealth(baseUrl, child) {
	for (let attempt = 0; attempt < 300; attempt += 1) {
		if (child.exitCode != null) throw new Error(`Server exited early with ${child.exitCode}`);
		try {
			const response = await fetch(`${baseUrl}/health`);
			if (response.ok) return;
		} catch {
			// Server is still starting.
		}
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	throw new Error("Server did not become healthy");
}

function collectNames(root) {
	const names = [];
	for (const entry of readdirSync(root, { withFileTypes: true })) {
		const path = join(root, entry.name);
		if (entry.isDirectory()) names.push(...collectNames(path));
		else names.push(path);
	}
	return names;
}

const root = mkdtempSync(join(tmpdir(), "inno-stream-upload-"));
const home = join(root, "runtime");
const workspace = join(root, "workspace");
const configDir = join(home, "config");
mkdirSync(configDir, { recursive: true });
mkdirSync(workspace, { recursive: true });
const configPath = join(configDir, "config.json");
writeFileSync(configPath, JSON.stringify({
	defaultProvider: "test",
	defaultModel: "test-model",
	providers: {
		test: {
			baseUrl: "http://127.0.0.1:9",
			api: "openai-completions",
			apiKey: "placeholder",
			models: [{ id: "test-model", name: "Test Model", contextWindow: 4096, maxTokens: 512 }],
		},
	},
	memory: { l1Enabled: false, l2Enabled: false, l3Enabled: false },
	subagents: { enabled: false },
}), "utf-8");

const port = await freePort();
const baseUrl = `http://127.0.0.1:${port}`;
const child = spawn(process.execPath, [
	fileURLToPath(new URL("../dist/server.js", import.meta.url)),
	"--home", home,
	"--config", configPath,
	"--workspace", workspace,
	"--port", String(port),
], { cwd: workspace, stdio: ["ignore", "pipe", "pipe"] });

let stderr = "";
let stdout = "";
child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });

try {
	await waitForHealth(baseUrl, child);
	const workspaceResponse = await fetch(`${baseUrl}/api/workspaces`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ name: "Streaming Upload Smoke", isTemp: false }),
	});
	if (!workspaceResponse.ok) throw new Error(`Workspace creation failed: ${await workspaceResponse.text()}`);
	const created = await workspaceResponse.json();

	const small = Buffer.from("x,y\n1,2\n", "utf-8");
	const uploadResponse = await fetch(`${baseUrl}/api/workspace/upload-file?workspaceId=${encodeURIComponent(created.id)}&path=small.csv`, {
		method: "POST",
		headers: { "Content-Type": "text/csv" },
		body: small,
	});
	if (uploadResponse.status !== 201) throw new Error(`Small streaming upload failed: ${await uploadResponse.text()}`);
	const upload = await uploadResponse.json();
	assert(upload.uploaded?.[0]?.size === small.length, "Uploaded size does not match streamed bytes");

	const rawResponse = await fetch(`${baseUrl}/api/l2/raw/upload?fileName=raw.csv`, {
		method: "POST",
		headers: { "Content-Type": "text/csv" },
		body: small,
	});
	if (rawResponse.status !== 201) throw new Error(`Raw streaming upload failed: ${await rawResponse.text()}`);

	const skillBytes = Buffer.from("---\nname: stream-smoke\ndescription: streamed test skill\n---\n\n# Test\n", "utf-8");
	const skillResponse = await fetch(`${baseUrl}/api/skills/upload?fileName=stream-smoke.md`, {
		method: "POST",
		headers: { "Content-Type": "text/markdown" },
		body: skillBytes,
	});
	if (skillResponse.status !== 201) throw new Error(`Skill streaming upload failed: ${await skillResponse.text()}`);

	const oversized = Buffer.alloc(25 * 1024 * 1024 + 1);
	const blockedResponse = await fetch(`${baseUrl}/api/workspace/upload-file?workspaceId=${encodeURIComponent(created.id)}&path=too-large.xlsx`, {
		method: "POST",
		headers: { "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" },
		body: oversized,
	});
	assert(blockedResponse.status === 413, `Oversized XLSX was not rejected: ${blockedResponse.status}`);
	const blocked = await blockedResponse.json();
	assert(blocked.maxBytes === 25 * 1024 * 1024, "Server returned the wrong XLSX limit");
	assert(!collectNames(root).some((path) => path.includes(".upload-") && path.endsWith(".tmp")), "Partial upload file was left behind");

	console.log(JSON.stringify({ ok: true, checks: 6, csvLimitMb: 50, excelLimitMb: 25 }));
} catch (error) {
	throw new Error(`${error instanceof Error ? error.message : String(error)}\nserver stdout:\n${stdout}\nserver stderr:\n${stderr}`);
} finally {
	if (child.exitCode == null) {
		child.kill();
		await Promise.race([
			new Promise((resolve) => child.once("exit", resolve)),
			new Promise((resolve) => setTimeout(resolve, 2_000)),
		]);
	}
	rmSync(root, { recursive: true, force: true });
}
