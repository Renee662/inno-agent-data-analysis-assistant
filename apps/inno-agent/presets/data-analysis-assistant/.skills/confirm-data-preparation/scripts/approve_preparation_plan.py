#!/usr/bin/env python3
"""Finalize item-level data-preparation choices without modifying any dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from file_utils import sha256_file as file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate final item-level choices and create an executable preparation plan."
    )
    parser.add_argument("--draft", required=True, help="Path to data-preparation-plan.json")
    parser.add_argument("--decisions", required=True, help="Path to the user's JSON decision record")
    parser.add_argument("--output-dir", required=True, help="Directory for approved plan files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing approved outputs in the output directory",
    )
    return parser.parse_args()


def fail(message: str, code: int = 2) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(code)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        fail(f"{label} file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Cannot read {label} JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} JSON must contain an object at its root")
    return value


def normalize_decisions(document: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    raw = document.get("decisions")
    if not isinstance(raw, dict):
        fail("Decision file must contain a decisions object")

    normalized: dict[str, dict[str, Any]] = {}
    for decision_id, value in raw.items():
        if not isinstance(decision_id, str) or not decision_id.strip():
            fail("Every decision ID must be a non-empty string")
        if isinstance(value, str):
            choice = value
            note = ""
            execution_parameters: dict[str, Any] = {}
        elif isinstance(value, dict):
            choice = value.get("choice")
            note = value.get("note", "")
            execution_parameters = value.get(
                "execution_parameters", value.get("parameters", {})
            )
        else:
            fail(f"Decision {decision_id!r} must be a choice string or an object")
        if not isinstance(choice, str) or not choice:
            fail(f"Decision {decision_id!r} is missing a choice")
        if not isinstance(note, str):
            fail(f"Decision {decision_id!r} note must be text")
        if not isinstance(execution_parameters, dict):
            fail(f"Decision {decision_id!r} execution_parameters must be an object")
        normalized[decision_id] = {
            "choice": choice,
            "note": note,
            "execution_parameters": execution_parameters,
        }

    return normalized, "user-questionnaire"


def collect_pending(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pending = draft.get("pending_decisions")
    if not isinstance(pending, list):
        fail("Draft plan must contain a pending_decisions array")
    result: dict[str, dict[str, Any]] = {}
    for item in pending:
        if not isinstance(item, dict):
            fail("Every pending decision must be an object")
        decision_id = item.get("id")
        options = item.get("options")
        if not isinstance(decision_id, str) or not decision_id:
            fail("Every pending decision must have a non-empty id")
        if decision_id in result:
            fail(f"Duplicate pending decision ID: {decision_id}")
        if item.get("status") != "pending":
            fail(f"Decision {decision_id!r} is listed as pending but has a different status")
        if not isinstance(options, list) or not options or not all(isinstance(x, str) for x in options):
            fail(f"Decision {decision_id!r} must provide a non-empty string options array")
        result[decision_id] = item
    return result


def apply_resolutions(value: Any, resolutions: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, list):
        return [apply_resolutions(item, resolutions) for item in value]
    if not isinstance(value, dict):
        return value

    resolved = {key: apply_resolutions(item, resolutions) for key, item in value.items()}
    decision_id = value.get("id")
    if decision_id in resolutions and value.get("status") == "pending":
        resolved["status"] = "user-confirmed"
        resolved["selected_option"] = resolutions[decision_id]["choice"]
        if resolutions[decision_id]["note"]:
            resolved["user_note"] = resolutions[decision_id]["note"]
        if resolutions[decision_id]["execution_parameters"]:
            resolved["execution_parameters"] = resolutions[decision_id][
                "execution_parameters"
            ]
    return resolved


def atomic_write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"Output already exists: {path}. Use --overwrite only after explicit approval.")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    )
    try:
        with handle:
            handle.write(content)
        os.replace(handle.name, path)
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_markdown(approved: dict[str, Any]) -> str:
    approval = approved["approval"]
    missingness = approved.get("missingness_bias_screen", {})
    conclusion_contract = approved.get("missingness_conclusion_contract", {})
    lines = [
        "# 已批准的数据处理方案",
        "",
        f"- 状态：`{approved['status']}`",
        f"- 批准时间：`{approval['approved_at']}`",
        f"- 批准主体：`{approval['confirmed_by']}`",
        f"- 源方案 SHA-256：`{approval['source_plan_sha256']}`",
        "- 原始数据未改动：`true`",
        "- 数据处理已执行：`false`",
        f"- 完整案例情景：保留 {missingness.get('complete_case_rows', '—')} 行，排除 {missingness.get('complete_case_excluded_rows', '—')} 行",
        f"- 缺失值结论范围：`{conclusion_contract.get('scope', '—')}`",
        "",
        "## 用户确认的决定",
        "",
        "| 决定 ID | 主题 | 建议 | 用户选择 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for item in approved["approved_decisions"]:
        lines.append(
            "| {id} | {topic} | {recommendation} | {choice} | {note} |".format(
                id=markdown_escape(item.get("id", "")),
                topic=markdown_escape(item.get("topic", "")),
                recommendation=markdown_escape(item.get("recommendation", "")),
                choice=markdown_escape(item.get("selected_option", "")),
                note=markdown_escape(item.get("user_note", "")),
            )
        )
    lines.extend(
        [
            "",
            "## 边界说明",
            "",
            "本文件只固化用户批准的处理决定。尚未执行删除、填补、编码、缩放或其他数据变换，也未生成 `cleaned-data.csv`。",
            "缺失组与非缺失组比较只描述样本构成，不能证明缺失机制；除非另有模型级敏感性证据，结论仅适用于实际进入模型的样本。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    draft_path = Path(args.draft).resolve()
    decisions_path = Path(args.decisions).resolve()
    output_dir = Path(args.output_dir).resolve()
    draft = read_json(draft_path, "draft plan")
    decision_document = read_json(decisions_path, "decision")
    if draft.get("status") != "draft":
        fail("Source plan status must be 'draft'")
    if draft.get("requires_user_confirmation") is not True:
        fail("Source plan must require user confirmation")
    if draft.get("raw_data_unchanged") is not True:
        fail("Source plan must state raw_data_unchanged: true")

    pending = collect_pending(draft)
    choices, confirmed_by = normalize_decisions(decision_document)
    pending_ids = set(pending)
    choice_ids = set(choices)
    missing = sorted(pending_ids - choice_ids)
    unknown = sorted(choice_ids - pending_ids)
    if missing:
        fail("Missing choices for: " + ", ".join(missing))
    if unknown:
        fail("Unknown decision IDs: " + ", ".join(unknown))

    for decision_id, choice_record in choices.items():
        allowed = pending[decision_id]["options"]
        if choice_record["choice"] not in allowed:
            fail(
                f"Invalid choice for {decision_id!r}: {choice_record['choice']!r}; "
                f"allowed choices are {allowed}"
            )

    approved_at = datetime.now().astimezone().isoformat()
    approved = apply_resolutions(copy.deepcopy(draft), choices)
    approved_decisions = [
        apply_resolutions(copy.deepcopy(pending[decision_id]), choices)
        for decision_id in pending
    ]
    approved["status"] = "approved"
    approved["requires_user_confirmation"] = False
    approved["raw_data_unchanged"] = True
    approved["data_preparation_executed"] = False
    approved["pending_decisions"] = []
    approved["approved_decisions"] = approved_decisions
    approved["approval"] = {
        "confirmed_by": confirmed_by,
        "approved_at": approved_at,
        "confirmation_method": "structured-item-questionnaire",
        "whole_plan_reconfirmation_required": False,
        "source_plan": str(draft_path),
        "source_plan_sha256": file_sha256(draft_path),
        "decision_record": str(decisions_path),
        "decision_record_sha256": file_sha256(decisions_path),
    }

    json_path = output_dir / "approved-data-preparation-plan.json"
    md_path = output_dir / "approved-data-preparation-plan.md"
    json_content = json.dumps(approved, ensure_ascii=False, indent=2) + "\n"
    md_content = build_markdown(approved)
    atomic_write_text(json_path, json_content, args.overwrite)
    try:
        atomic_write_text(md_path, md_content, args.overwrite)
    except Exception:
        if not args.overwrite and json_path.exists():
            try:
                json_path.unlink()
            except OSError:
                pass
        raise

    print(
        json.dumps(
            {
                "ok": True,
                "status": "approved",
                "approved_decisions": len(approved_decisions),
                "raw_data_unchanged": True,
                "data_preparation_executed": False,
                "outputs": [str(md_path), str(json_path)],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
