#!/usr/bin/env node

import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { prepareQuestionnairePresentation } from "../dist/agent/question-presentation.js";
import { recordQuestionApprovals } from "../dist/agent/approval-record.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const workspace = mkdtempSync(join(tmpdir(), "inno-question-presentation-"));
mkdirSync(join(workspace, "conversations", "c1", "work", "analysis-plan"), { recursive: true });
mkdirSync(join(workspace, "conversations", "c1", "work", "figures"), { recursive: true });
writeFileSync(join(workspace, "conversations", "c1", "work", "analysis-plan", "analysis-task.json"), JSON.stringify({ title: "test" }));
writeFileSync(join(workspace, "conversations", "c1", "work", "figures", "plot.png"), "png");

const missingKind = prepareQuestionnairePresentation({ questions: [{ header: "任务卡确认", question: "分析任务卡是否正确？", options: [{ label: "正确", description: "继续" }] }] }, workspace, "conversations/c1/work/analysis-plan/analysis-task.json");
assert(!missingKind.ok, "Data-analysis questions without an explicit kind must be rejected");

const taskCard = prepareQuestionnairePresentation({ questions: [{ questionKind: "task-card", header: "任务卡确认", question: "分析任务卡是否正确？", approvalAction: "approve-analysis-task", approvalArtifactPath: "conversations/c1/work/analysis-plan/analysis-task.json", options: [{ label: "正确", description: "继续", approvalDecision: "approve" }, { label: "修改", description: "返回修改", approvalDecision: "revise" }] }] }, workspace, "conversations/c1/work/analysis-plan/analysis-task.json");
assert(taskCard.ok && taskCard.params.questions[0].documentPath.endsWith("analysis-task.json"), "Task card must be attached automatically");
assert(taskCard.params.questions[0].approvalArtifactSha256?.length === 64, "Approval artifact hash must be computed by the server");

const missingImage = prepareQuestionnairePresentation({ questions: [{ questionKind: "observation", header: "观察图形", question: "从比例图看有什么关系？", options: [{ label: "A", description: "A" }] }] }, workspace);
assert(!missingImage.ok, "Observation questions without an image must be rejected");

const withImage = prepareQuestionnairePresentation({ questions: [{ questionKind: "observation", header: "观察", question: "请选择", imagePath: "conversations/c1/work/figures/plot.png", options: [{ label: "A", description: "A" }] }] }, workspace);
assert(withImage.ok && withImage.params.questions[0].imagePath.includes("figures/plot.png"), "Observation image must be retained");

const recorded = recordQuestionApprovals(taskCard.params, { questionId: "q1", cancelled: false, answers: [{ questionIndex: 0, question: "分析任务卡是否正确？", kind: "option", answer: "正确" }] }, { workspaceDir: workspace, sessionId: "s1", questionId: "q1", source: "web-question-dialog" });
assert(recorded.approvalRecords?.length === 1, "Approving UI response must create one receipt");

console.log(JSON.stringify({ ok: true, checks: 6 }));
