#!/usr/bin/env node

import { userFacingTextFromAssistantMessage } from "../dist/agent/user-facing-output.js";

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

assertEqual(
  userFacingTextFromAssistantMessage({
    role: "assistant",
    stopReason: "toolUse",
    content: [
      { type: "text", text: "我需要检查执行脚本如何分派这些决策。" },
      { type: "toolCall", name: "read", arguments: { path: "script.py" } },
    ],
  }),
  "",
  "tool-use narration is hidden",
);

assertEqual(
  userFacingTextFromAssistantMessage({
    role: "assistant",
    stopReason: "toolUse",
    content: [
      { type: "text", text: "数据中违约者约占两成。下面需要确认研究问题和变量角色，避免把预测目标误当作普通特征。" },
      { type: "toolCall", name: "ask_user_question", arguments: { questions: [] } },
    ],
  }),
  "数据中违约者约占两成。下面需要确认研究问题和变量角色，避免把预测目标误当作普通特征。",
  "core questionnaire explanation is retained",
);

assertEqual(
  userFacingTextFromAssistantMessage({
    role: "assistant",
    stopReason: "stop",
    content: [
      { type: "thinking", thinking: "internal" },
      { type: "text", text: "处理已停止：当前文件需要专门的双表头读取流程。" },
    ],
  }),
  "处理已停止：当前文件需要专门的双表头读取流程。",
  "terminal answer is retained",
);

assertEqual(
  userFacingTextFromAssistantMessage({
    role: "assistant",
    stopReason: "error",
    content: [{ type: "text", text: "partial provider error" }],
  }),
  "",
  "provider error text is handled by the error channel",
);

console.log(JSON.stringify({ ok: true, cases: 4 }));
