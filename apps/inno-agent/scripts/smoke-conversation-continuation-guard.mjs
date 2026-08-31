#!/usr/bin/env node

import {
  createContinuationGuardState,
  isBenignTerminalContinuationError,
  noteContinuationGuardEvent,
} from "../dist/agent/conversation-continuation-guard.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const completed = createContinuationGuardState();
noteContinuationGuardEvent(completed, {
  type: "message_end",
  message: { role: "assistant", stopReason: "stop" },
});
noteContinuationGuardEvent(completed, {
  type: "compaction_end",
  reason: "overflow",
  aborted: false,
  willRetry: true,
  result: { summary: "ok" },
});
assert(
  isBenignTerminalContinuationError(new Error("Cannot continue from message role: assistant"), completed),
  "The exact completed-response overflow sequence must be suppressed",
);

const actualFailure = createContinuationGuardState();
noteContinuationGuardEvent(actualFailure, {
  type: "message_end",
  message: { role: "assistant", stopReason: "error" },
});
noteContinuationGuardEvent(actualFailure, {
  type: "compaction_end",
  reason: "overflow",
  aborted: false,
  willRetry: true,
  result: { summary: "retry" },
});
assert(
  !isBenignTerminalContinuationError(new Error("Cannot continue from message role: assistant"), actualFailure),
  "A real model error must not be hidden",
);
assert(
  !isBenignTerminalContinuationError(new Error("network failed"), completed),
  "Unrelated failures must not be hidden",
);

console.log(JSON.stringify({ ok: true, checks: 3 }));
