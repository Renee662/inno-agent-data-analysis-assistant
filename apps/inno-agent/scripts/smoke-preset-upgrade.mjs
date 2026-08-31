#!/usr/bin/env node

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ensurePresetCached, instantiatePreset } from "../dist/presets/preset-store.js";
import { WorkspaceRegistry } from "../dist/workspace/workspace-registry.js";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const tempRoot = mkdtempSync(join(tmpdir(), "inno-preset-upgrade-"));
const expectedVersion = JSON.parse(
  readFileSync(join(appRoot, "presets", "data-analysis-assistant", "preset.json"), "utf8"),
).version;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function filesUnder(root) {
  if (!existsSync(root)) return [];
  const found = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) found.push(...filesUnder(path));
    else if (entry.isFile()) found.push(path);
  }
  return found;
}

try {
  const dataDir = join(tempRoot, "data");
  const workspaceRoot = join(tempRoot, "workspace");
  const cacheRoot = join(dataDir, "preset-cache");
  const cacheDir = join(cacheRoot, "data-analysis-assistant");
  mkdirSync(cacheDir, { recursive: true });
  writeFileSync(
    join(cacheDir, "preset.json"),
    JSON.stringify({
      id: "data-analysis-assistant",
      name: "数据分析助手",
      description: "legacy cache",
      version: 1,
    }),
  );
  writeFileSync(join(cacheDir, "agent.md"), "legacy cached agent\n");

  const paths = {
    codeDir: appRoot,
    presetCacheDir: cacheRoot,
  };
  const unusedSource = {
    async downloadItem() { throw new Error("remote source must not be used"); },
  };
  await ensurePresetCached(paths, unusedSource, "data-analysis-assistant");
  const cachedMeta = JSON.parse(readFileSync(join(cacheDir, "preset.json"), "utf8"));
  assert(cachedMeta.version === expectedVersion, "Bundled version did not upgrade the stale cache");
  assert(readFileSync(join(cacheDir, "agent.md"), "utf8") !== "legacy cached agent\n", "Cached files were not refreshed");

  const registry = new WorkspaceRegistry(workspaceRoot, dataDir);
  registry.ensureBootstrapped();
  const first = instantiatePreset(paths, registry, "data-analysis-assistant");
  const workspaceDir = registry.resolveWorkspaceDir(first.id);
  assert(workspaceDir, "Preset workspace was not created");
  const markerPath = join(workspaceDir, ".inno", "preset-installation.json");
  assert(JSON.parse(readFileSync(markerPath, "utf8")).version === expectedVersion, "Initial preset version was not recorded");

  writeFileSync(join(workspaceDir, "agent.md"), "local edited agent\n");
  writeFileSync(markerPath, JSON.stringify({ presetId: "data-analysis-assistant", version: 1 }));
  mkdirSync(join(workspaceDir, "conversations"), { recursive: true });
  writeFileSync(join(workspaceDir, "conversations", "keep.txt"), "keep\n");

  instantiatePreset(paths, registry, "data-analysis-assistant");
  assert(JSON.parse(readFileSync(markerPath, "utf8")).version === expectedVersion, "Existing workspace was not upgraded");
  assert(readFileSync(join(workspaceDir, "conversations", "keep.txt"), "utf8") === "keep\n", "Conversation artifacts were changed")
  const backups = filesUnder(join(workspaceDir, ".inno", "preset-backups"));
  const preservedEdit = backups.some((path) => readFileSync(path, "utf8") === "local edited agent\n");
  assert(preservedEdit, "Conflicting managed file was not backed up before upgrade");
  console.log(JSON.stringify({
    ok: true,
    cachedVersion: cachedMeta.version,
    installedVersion: JSON.parse(readFileSync(markerPath, "utf8")).version,
    backups: backups.length,
  }));
} finally {
  rmSync(tempRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
