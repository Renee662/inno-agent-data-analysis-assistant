import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { motion } from "motion/react";
import { AlertCircle, Check, CircleHelp, LoaderCircle } from "lucide-react";
import type { PendingQuestion, QuestionAnswer, QuestionData, QuestionnaireResult } from "../types/chat.js";
import { chatStore } from "../stores/chat-store.js";
import { getWorkspaceFile, workspaceFileUrl } from "../api/workspace.js";

type TaskCardRecord = Record<string, unknown>;

function textValue(value: unknown): string | undefined {
	return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function TaskCardPreview({ q, workspaceId }: { q: QuestionData; workspaceId: string | null }) {
	const [content, setContent] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let active = true;
		setContent(null);
		setError(null);
		if (!q.documentPath) return () => { active = false; };
		void getWorkspaceFile(q.documentPath, workspaceId ?? undefined, true)
			.then((file) => { if (active) setContent(file.content ?? ""); })
			.catch(() => { if (active) setError("任务卡读取失败，请刷新后重试。"); });
		return () => { active = false; };
	}, [q.documentPath, workspaceId]);

	if (!q.documentPath) return null;
	if (error) return <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>;
	if (content === null) {
		return <div className="flex items-center gap-2 rounded-lg border border-[var(--inno-border)] px-3 py-3 text-xs text-[var(--inno-text-muted)]"><LoaderCircle className="h-4 w-4 animate-spin" />正在载入任务卡…</div>;
	}

	if (!q.documentPath.toLowerCase().endsWith(".json")) {
		return (
			<section className="max-h-72 overflow-y-auto rounded-lg border border-[var(--inno-border)] bg-[var(--inno-surface-muted)] p-3 text-sm">
				<markdown-artifact content={content} />
			</section>
		);
	}

	let card: TaskCardRecord;
	try {
		card = JSON.parse(content) as TaskCardRecord;
	} catch {
		return <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700">任务卡 JSON 格式无效，暂不能确认。</div>;
	}
	const metadata = card.variable_metadata && typeof card.variable_metadata === "object"
		? card.variable_metadata as Record<string, TaskCardRecord>
		: {};
	const labelFor = (column: unknown) => {
		if (typeof column !== "string") return String(column ?? "");
		return textValue(metadata[column]?.display_name) ?? column;
	};
	const predictors = Array.isArray(card.predictors) ? card.predictors.map(labelFor) : [];
	const controls = Array.isArray(card.controls) ? card.controls.map(labelFor) : [];
	const rows = [
		["研究问题", textValue(card.research_question)],
		["分析单位", textValue(card.unit_of_analysis) ?? "尚待确认"],
		["结果变量", labelFor(card.outcome)],
		["预测变量", predictors.length ? `${predictors.join("、")}（共 ${predictors.length} 项）` : "未指定"],
		["控制变量", controls.length ? controls.join("、") : "无"],
	].filter((row) => row[1]);

	return (
		<section className="overflow-hidden rounded-lg border border-[var(--inno-border)] bg-[var(--inno-surface-muted)]">
			<div className="border-b border-[var(--inno-border)] bg-[var(--inno-surface)] px-3 py-2">
				<p className="text-xs font-semibold text-[var(--inno-text)]">{q.documentTitle || "分析任务卡"}</p>
				<p className="mt-0.5 text-[11px] text-[var(--inno-text-muted)]">{textValue(card.report_title) ?? textValue(card.title)}</p>
			</div>
			<div className="max-h-72 space-y-2 overflow-y-auto p-3 text-xs leading-5">
				{textValue(card.dataset_summary) ? <p className="text-[var(--inno-text-muted)]">{textValue(card.dataset_summary)}</p> : null}
				<dl className="grid grid-cols-[5rem_minmax(0,1fr)] gap-x-2 gap-y-1.5">
					{rows.map(([label, value]) => <div key={label} className="contents"><dt className="font-medium text-[var(--inno-text-muted)]">{label}</dt><dd className="text-[var(--inno-text)]">{value}</dd></div>)}
				</dl>
			</div>
			{q.documentCaption ? <p className="border-t border-[var(--inno-border)] px-3 py-2 text-[11px] text-[var(--inno-text-muted)]">{q.documentCaption}</p> : null}
		</section>
	);
}

function OptionRow({
	label,
	description,
	selected,
	multi,
	onSelect,
	onFocus,
}: {
	label: string;
	description: string;
	selected: boolean;
	multi: boolean;
	onSelect: () => void;
	onFocus: () => void;
}) {
	return (
		<button
			type="button"
			aria-pressed={selected}
			className={`flex w-full items-start gap-2.5 rounded-md border px-3 py-2 text-left text-[13px] transition-colors ${
				selected
					? "border-[var(--inno-accent)] bg-[var(--inno-accent-soft)] text-[var(--inno-text)]"
					: "border-[var(--inno-border)] bg-[var(--inno-surface)] text-[var(--inno-text)] hover:border-[var(--inno-border-strong)] hover:bg-[var(--inno-surface-muted)]"
			}`}
			onClick={onSelect}
			onMouseEnter={onFocus}
			onFocus={onFocus}
		>
			<span className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center border border-[var(--inno-border-strong)] ${multi ? "rounded-sm" : "rounded-full"}`}>
				{selected ? (
					<span className={`block h-2 w-2 bg-[var(--inno-accent)] ${multi ? "rounded-sm" : "rounded-full"}`} />
				) : null}
			</span>
			<span className="min-w-0 flex-1">
				<span className="font-medium">{label}</span>
				{description ? <span className="mt-0.5 block text-xs leading-5 text-[var(--inno-text-muted)]">{description}</span> : null}
			</span>
		</button>
	);
}

function QuestionTab({
	q,
	questionIndex,
	answer,
	onAnswer,
	onDismiss,
	focusedOption,
	setFocusedOption,
	customDraft,
	onCustomDraftChange,
	workspaceId,
}: {
	q: QuestionData;
	questionIndex: number;
	answer: QuestionAnswer | undefined;
	onAnswer: (answer: QuestionAnswer) => void;
	onDismiss: () => void;
	focusedOption: number;
	setFocusedOption: (index: number) => void;
	customDraft: string;
	onCustomDraftChange: (text: string) => void;
	workspaceId: string | null;
}) {
	const { t } = useTranslation();
	const [imageFailed, setImageFailed] = useState(false);
	const isMulti = q.multiSelect === true;
	const hasPreview = q.options.some((option) => option.preview);
	const hasOptionAnswer = answer?.kind === "option" || answer?.kind === "multi";
	const selectedLabels = new Set(
		answer?.kind === "multi"
			? (answer.selected ?? [])
			: answer?.kind === "option" && answer.answer
				? [answer.answer]
				: [],
	);

	const handleOptionClick = (label: string) => {
		const notes = customDraft.trim() || undefined;
		if (isMulti) {
			const next = new Set(selectedLabels);
			if (next.has(label)) next.delete(label);
			else next.add(label);
			onAnswer({
				questionIndex,
				question: q.question,
				kind: "multi",
				answer: null,
				selected: Array.from(next),
				notes,
			});
			return;
		}

		if (selectedLabels.has(label)) {
			onAnswer(
				notes
					? { questionIndex, question: q.question, kind: "custom", answer: notes }
					: { questionIndex, question: q.question, kind: "option", answer: null },
			);
			return;
		}

		onAnswer({
			questionIndex,
			question: q.question,
			kind: "option",
			answer: label,
			notes,
			preview: q.options.find((option) => option.label === label)?.preview,
		});
	};

	const handleCustomChange = (text: string) => {
		onCustomDraftChange(text);
		const trimmed = text.trim();
		if (hasOptionAnswer && answer) {
			onAnswer({ ...answer, notes: trimmed || undefined });
			return;
		}
		if (isMulti) return;
		if (trimmed) {
			onAnswer({ questionIndex, question: q.question, kind: "custom", answer: trimmed });
		} else if (answer?.kind === "custom") {
			onAnswer({ questionIndex, question: q.question, kind: "custom", answer: null });
		}
	};

	const preview = hasPreview ? q.options[focusedOption]?.preview : undefined;
	const imageUrl = q.imagePath
		? workspaceFileUrl(q.imagePath, workspaceId ?? undefined)
		: undefined;

	useEffect(() => {
		setImageFailed(false);
	}, [imageUrl]);

	return (
		<div className="space-y-3">
			<TaskCardPreview q={q} workspaceId={workspaceId} />
			{imageUrl && !imageFailed ? (
				<figure className="overflow-hidden rounded-lg border border-[var(--inno-border)] bg-[var(--inno-surface-muted)]">
					<a
						href={imageUrl}
						target="_blank"
						rel="noreferrer"
						className="flex max-h-56 min-h-32 items-center justify-center bg-white p-2"
						title={t("question.openImage")}
					>
						<img
							src={imageUrl}
							alt={q.imageAlt || t("question.referenceImage")}
							className="block max-h-52 max-w-full object-contain"
							onError={() => setImageFailed(true)}
						/>
					</a>
					<figcaption className="border-t border-[var(--inno-border)] px-3 py-2 text-xs leading-5 text-[var(--inno-text-muted)]">
						{q.imageCaption || t("question.openImage")}
					</figcaption>
				</figure>
			) : null}

			<div>
				<p className="text-sm font-semibold leading-6 text-[var(--inno-text)]">{q.question}</p>
				<p className="mt-1 text-xs leading-5 text-[var(--inno-text-muted)]">
					{isMulti ? t("question.multiSelectHint") : t("question.singleSelectHint")}
				</p>
			</div>

			<div className={hasPreview ? "flex flex-col gap-3 lg:flex-row" : ""}>
				<div className={`space-y-2 ${hasPreview ? "lg:w-1/2" : ""}`}>
					{q.options.map((option, index) => (
						<OptionRow
							key={option.label}
							label={option.label}
							description={option.description}
							selected={selectedLabels.has(option.label)}
							multi={isMulti}
							onSelect={() => handleOptionClick(option.label)}
							onFocus={() => setFocusedOption(index)}
						/>
					))}

					<label className="flex flex-col gap-1.5 pt-1">
						<span className="text-xs font-medium text-[var(--inno-text-muted)]">
							{hasOptionAnswer ? t("question.optionalNote") : t("question.customAnswer")}
						</span>
						<textarea
							rows={2}
							className="min-w-0 resize-y rounded-md border border-[var(--inno-border)] bg-[var(--inno-surface)] px-2.5 py-2 text-[13px] text-[var(--inno-text)] focus-visible:border-[var(--inno-focus-border)] focus-visible:outline-none focus-visible:shadow-[var(--inno-ring)]"
							placeholder={
								isMulti && !hasOptionAnswer
									? t("question.multiNotePlaceholder")
									: hasOptionAnswer
										? t("question.optionalNotePlaceholder")
										: t("question.typeSomething")
							}
							value={customDraft}
							onChange={(event) => handleCustomChange(event.target.value)}
						/>
					</label>
				</div>

				{hasPreview && preview ? (
					<div className="rounded-md border border-[var(--inno-border)] bg-[var(--inno-surface-muted)] p-3 lg:w-1/2">
						<pre className="whitespace-pre-wrap font-mono text-xs text-[var(--inno-text)]">{preview}</pre>
					</div>
				) : null}
			</div>

			<button
				type="button"
				className="text-xs text-[var(--inno-text-subtle)] underline hover:text-[var(--inno-text-muted)]"
				onClick={onDismiss}
			>
				{t("question.chatAboutThis")}
			</button>
		</div>
	);
}

export function QuestionDialog({ pending, workspaceId }: { pending: PendingQuestion; workspaceId: string | null }) {
	const { t } = useTranslation();
	const { questionId, params } = pending;
	const questions = params.questions;
	const [activeTab, setActiveTab] = useState(0);
	const [answers, setAnswers] = useState<Map<number, QuestionAnswer>>(new Map());
	const [focusedOptions, setFocusedOptions] = useState<number[]>(questions.map(() => 0));
	const [customDrafts, setCustomDrafts] = useState<Map<number, string>>(new Map());
	const [isSubmitting, setIsSubmitting] = useState(false);
	const [submitError, setSubmitError] = useState("");

	const handleAnswer = useCallback((answer: QuestionAnswer) => {
		setSubmitError("");
		setAnswers((previous) => {
			const next = new Map(previous);
			if ((answer.kind === "custom" || answer.kind === "option") && answer.answer === null) {
				next.delete(answer.questionIndex);
			} else if (answer.kind === "multi" && (!answer.selected || answer.selected.length === 0)) {
				next.delete(answer.questionIndex);
			} else {
				next.set(answer.questionIndex, answer);
			}
			return next;
		});
	}, []);

	const handleDismiss = useCallback(async () => {
		setIsSubmitting(true);
		setSubmitError("");
		try {
			await chatStore.dismissQuestion(questionId);
		} catch (error) {
			setSubmitError(error instanceof Error ? error.message : t("question.submitError"));
			setIsSubmitting(false);
		}
	}, [questionId, t]);

	const currentAnswered = answers.has(activeTab);
	const isLast = activeTab === questions.length - 1;
	const allAnswered = questions.every((_, index) => answers.has(index));
	const unansweredCount = questions.length - answers.size;

	const handleClick = useCallback(async () => {
		if (!answers.has(activeTab)) return;
		if (!isLast) {
			setActiveTab((previous) => Math.min(previous + 1, questions.length - 1));
			return;
		}

		const firstUnanswered = questions.findIndex((_, index) => !answers.has(index));
		if (firstUnanswered >= 0) {
			setActiveTab(firstUnanswered);
			return;
		}

		const result: QuestionnaireResult = {
			answers: Array.from(answers.values()).sort((left, right) => left.questionIndex - right.questionIndex),
			cancelled: false,
		};
		setIsSubmitting(true);
		setSubmitError("");
		try {
			await chatStore.submitQuestionResponse(questionId, result);
		} catch (error) {
			setSubmitError(error instanceof Error ? error.message : t("question.submitError"));
			setIsSubmitting(false);
		}
	}, [activeTab, answers, isLast, questionId, questions, t]);

	const setFocusedForTab = useCallback((tab: number, optionIndex: number) => {
		setFocusedOptions((previous) => {
			const next = [...previous];
			next[tab] = optionIndex;
			return next;
		});
	}, []);

	const setCustomDraftForTab = useCallback((tab: number, value: string) => {
		setCustomDrafts((previous) => {
			const next = new Map(previous);
			next.set(tab, value);
			return next;
		});
	}, []);

	return (
		<motion.div
			className="flex justify-start"
			initial={{ opacity: 0, y: 16 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.3, ease: "easeOut" }}
		>
			<div className="w-full max-w-[88%] rounded-xl border border-[var(--inno-accent-soft)] bg-[var(--inno-surface)] px-4 py-4 shadow-sm sm:px-5">
				<div className="mb-3 flex items-start gap-2.5 border-b border-[var(--inno-border)] pb-3">
					<span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--inno-accent-soft)] text-[var(--inno-accent)]">
						<CircleHelp size={16} />
					</span>
					<div>
						<p className="text-sm font-semibold text-[var(--inno-text)]">{t("question.decisionTitle")}</p>
						<p className="mt-0.5 text-xs leading-5 text-[var(--inno-text-muted)]">{t("question.decisionDescription")}</p>
					</div>
				</div>

				{questions.length > 1 ? (
					<div className="mb-3 flex flex-wrap gap-1.5">
						{questions.map((question, index) => (
							<button
								type="button"
								key={`${question.header}-${index}`}
								className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
									activeTab === index
										? "bg-[var(--inno-accent-soft)] text-[var(--inno-accent)]"
										: "bg-[var(--inno-surface-muted)] text-[var(--inno-text-muted)] hover:bg-[var(--inno-surface-muted)]"
								}`}
								onClick={() => setActiveTab(index)}
							>
								{answers.has(index) ? <Check size={12} /> : null}
								{question.header}
							</button>
						))}
					</div>
				) : null}

				<fieldset disabled={isSubmitting}>
					<QuestionTab
						q={questions[activeTab]}
						questionIndex={activeTab}
						answer={answers.get(activeTab)}
						onAnswer={handleAnswer}
						onDismiss={() => { void handleDismiss(); }}
						focusedOption={focusedOptions[activeTab]}
						setFocusedOption={(index) => setFocusedForTab(activeTab, index)}
						customDraft={customDrafts.get(activeTab) ?? ""}
						onCustomDraftChange={(value) => setCustomDraftForTab(activeTab, value)}
						workspaceId={workspaceId}
					/>
				</fieldset>

				{submitError ? (
					<div className="mt-3 flex items-start gap-2 rounded-md border border-[var(--inno-danger-border)] bg-[var(--inno-danger-bg)] px-3 py-2 text-xs text-[var(--inno-danger)]">
						<AlertCircle className="mt-0.5 shrink-0" size={14} />
						<span>{t("question.submitErrorDetail", { error: submitError })}</span>
					</div>
				) : null}

				<div className="mt-3 flex items-center justify-end gap-2">
					{questions.length > 1 ? (
						<span className="text-xs text-[var(--inno-text-subtle)]">
							{t("question.progress", { current: activeTab + 1, total: questions.length })}
						</span>
					) : null}
					<button
						type="button"
						className={`inline-flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
							currentAnswered && !isSubmitting
								? "inno-primary-button"
								: "cursor-not-allowed bg-[var(--inno-surface-muted)] text-[var(--inno-text-subtle)]"
						}`}
						disabled={!currentAnswered || isSubmitting}
						onClick={() => { void handleClick(); }}
					>
						{isSubmitting ? <LoaderCircle className="animate-spin" size={14} /> : null}
						{isSubmitting
							? t("question.submitting")
							: isLast && allAnswered
								? t("question.submit")
								: isLast
									? t("question.completeMissing", { count: unansweredCount })
									: t("question.submitNext")}
					</button>
				</div>
			</div>
		</motion.div>
	);
}
