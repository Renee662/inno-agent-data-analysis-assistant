#!/usr/bin/env node

import { spawn } from "node:child_process";
import {
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(appRoot, "..", "..");
const tempRoot = mkdtempSync(join(tmpdir(), "inno-conversation-inputs-"));
const homeDir = join(tempRoot, "runtime");
const dataDir = join(homeDir, "data");
const workspaceDir = join(tempRoot, "workspace");
const cacheDir = join(dataDir, "preset-cache", "data-analysis-assistant");
const configDir = join(homeDir, "config");
// Never read the user's runtime config. The isolated server uses only the
// repository's placeholder-only example configuration.
const sourceConfig = join(repoRoot, "config.example.json");
const port = 31_000 + Math.floor(Math.random() * 1_000);
let child;
let serverOutput = "";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitForHealth() {
  // Loading the full server dependency graph can exceed 20 seconds on a cold
  // Windows filesystem. Keep the same 250 ms probe cadence but allow 60 seconds.
  for (let attempt = 0; attempt < 240; attempt += 1) {
    if (child?.exitCode !== null && child?.exitCode !== undefined) {
      throw new Error(`Isolated smoke-test server exited with code ${child.exitCode}`);
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return;
    } catch {
      // Server is still starting.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
  throw new Error("Isolated smoke-test server did not become healthy");
}

async function request(path, init) {
  const response = await fetch(`http://127.0.0.1:${port}${path}`, init);
  const payload = await response.json();
  return { response, payload };
}

try {
  mkdirSync(configDir, { recursive: true });
  mkdirSync(workspaceDir, { recursive: true });
  cpSync(sourceConfig, join(configDir, "config.json"));
  cpSync(join(appRoot, "presets", "data-analysis-assistant"), cacheDir, {
    recursive: true,
  });

  child = spawn(
    process.execPath,
    [
      join(appRoot, "dist", "server.js"),
      "--home",
      homeDir,
      "--workspace",
      workspaceDir,
      "--port",
      String(port),
    ],
    {
      cwd: repoRoot,
      env: { ...process.env, INNO_PORT: String(port) },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    },
  );
  child.stdout.on("data", (chunk) => { serverOutput += String(chunk).slice(-4_000); });
  child.stderr.on("data", (chunk) => { serverOutput += String(chunk).slice(-4_000); });
  child.on("error", (error) => { serverOutput += `\nspawn error: ${error.message}`; });
  child.on("exit", (code, signal) => { serverOutput += `\nserver exit: ${code ?? "null"}/${signal ?? "none"}`; });

  await waitForHealth();
  const created = await request("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ presetId: "data-analysis-assistant" }),
  });
  assert(created.response.status === 201, `Session creation failed: ${JSON.stringify(created.payload)}`);
  const { id: conversationId, workspaceId } = created.payload;
  assert(typeof conversationId === "string", "Session did not return an id");
  assert(workspaceId === "preset-data-analysis-assistant", "Unexpected preset workspace id");

  const uploadBytes = Buffer.from("x,y\n1,2\n");
  const uploadPath = `/api/workspace/upload-file?workspaceId=${encodeURIComponent(workspaceId)}&conversationId=${encodeURIComponent(conversationId)}&path=data.csv`;
  const first = await request(uploadPath, {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: uploadBytes,
  });
  const second = await request(uploadPath, {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: uploadBytes,
  });
  assert(first.response.status === 201 && second.response.status === 201, "Upload failed");
  const firstPath = first.payload.uploaded?.[0]?.path;
  const secondPath = second.payload.uploaded?.[0]?.path;
  assert(firstPath?.includes("/inputs/data.csv"), `Unexpected first upload path: ${firstPath}`);
  assert(secondPath?.includes("/inputs/data (2).csv"), `Upload was not versioned: ${secondPath}`);
  assert(firstPath !== secondPath, "Same-named uploads silently overwrote each other");

  const workspaceRoot = join(workspaceDir, ".presets", "data-analysis-assistant");
  assert(readFileSync(join(workspaceRoot, firstPath), "utf8") === "x,y\n1,2\n", "First upload changed");
  assert(readFileSync(join(workspaceRoot, secondPath), "utf8") === "x,y\n1,2\n", "Second upload missing");

  const unscoped = await request(`/api/workspace/upload-file?workspaceId=${encodeURIComponent(workspaceId)}&path=data.csv`, {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: uploadBytes,
  });
  assert(unscoped.response.status === 400, "Unscoped data-analysis upload was not rejected");
  console.log(JSON.stringify({ ok: true, firstPath, secondPath }));
} catch (error) {
  throw new Error(
    `${error instanceof Error ? error.message : String(error)}`
    + `${child ? "" : " (server not started)"}`
    + `${serverOutput ? `\nServer output:\n${serverOutput.slice(-8_000)}` : ""}`,
  );
} finally {
  if (child && child.exitCode === null) {
    child.kill();
    await Promise.race([
      new Promise((resolveExit) => child.once("exit", resolveExit)),
      new Promise((resolveWait) => setTimeout(resolveWait, 3_000)),
    ]);
  }
  rmSync(tempRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
