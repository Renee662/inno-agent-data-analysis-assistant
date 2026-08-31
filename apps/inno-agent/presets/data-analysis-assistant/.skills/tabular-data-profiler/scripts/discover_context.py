#!/usr/bin/env python3
"""Discover local files that may explain a tabular dataset without changing inputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml", ".rtf"}
DOCUMENT_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
}
TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | DOCUMENT_SUFFIXES | TABULAR_SUFFIXES
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "analysis",
    "outputs",
    "runtime",
    "work",
    "__pycache__",
}
IGNORED_FILE_NAMES = {
    "agent.md",
    "preset.json",
    "analysis-task.json",
    "approved-data-preparation-plan.json",
    "approved-model-specification.json",
    "data-preparation-log.json",
    "model-summary.json",
    "analysis-run-log.json",
    "report-manifest.json",
    "final-report.html",
}
MAX_EXCERPT_CHARS = 8_000
MAX_FILES = 30
MAX_DEPTH = 2

CONTEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("codebook", re.compile(r"code.?book|data.?dictionary|variable.?list|字段|变量|字典|编码", re.I)),
    ("readme", re.compile(r"read.?me|about|overview|description|介绍|说明|简介", re.I)),
    ("questionnaire", re.compile(r"questionnaire|survey|instrument|问卷|量表|调查", re.I)),
    ("metadata", re.compile(r"meta.?data|schema|label|mapping|元数据|标签|映射", re.I)),
    ("source-note", re.compile(r"source|citation|reference|来源|出处|引用", re.I)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=Path, help="Primary dataset table")
    parser.add_argument("--profile", required=True, type=Path, help="Generated data-profile.json")
    parser.add_argument(
        "--context-root",
        type=Path,
        help="Directory to scan; defaults to the table's parent directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for dataset-context.json and dataset-context.md",
    )
    return parser.parse_args()


def read_profile(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read profile JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), list):
        raise ValueError("Profile JSON does not contain a profiles array")
    return payload


def profile_columns(profile: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for sheet in profile.get("profiles", []):
        if not isinstance(sheet, dict):
            continue
        for column in sheet.get("columns", []):
            if isinstance(column, dict) and isinstance(column.get("column"), str):
                columns.append(column["column"])
    return list(dict.fromkeys(columns))


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_text_file(path: Path) -> tuple[str, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return "", f"read_error: {exc}"
    text = decode_text(raw[: 512_000])
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return text[:MAX_EXCERPT_CHARS], "extracted"


def extract_docx(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
    except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return "", f"docx_extract_error: {exc}"
    paragraphs: list[str] = []
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for paragraph in root.iter(f"{namespace}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
        value = "".join(runs).strip()
        if value:
            paragraphs.append(value)
    return "\n".join(paragraphs)[:MAX_EXCERPT_CHARS], "extracted"


def extract_tabular_context(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return extract_text_file(path)
    try:
        import pandas as pd
    except ImportError:
        return "", "requires_tabular_preview"
    try:
        workbook = pd.ExcelFile(path)
        sections: list[str] = []
        for sheet_name in workbook.sheet_names[:5]:
            frame = pd.read_excel(workbook, sheet_name=sheet_name, nrows=50)
            sections.append(f"[sheet: {sheet_name}]")
            sections.append(frame.iloc[:, :12].to_csv(index=False, sep="\t"))
        return "\n".join(sections)[:MAX_EXCERPT_CHARS], "extracted"
    except Exception as exc:  # noqa: BLE001 - surface parser error as evidence status
        return "", f"tabular_extract_error: {exc}"


def normalized_excerpt(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_EXCERPT_CHARS]


def category_for(path: Path) -> tuple[str, int]:
    haystack = f"{path.stem} {path.parent.name}"
    for category, pattern in CONTEXT_PATTERNS:
        if pattern.search(haystack):
            return category, 5
    return "other", 0


def column_mentions(text: str, columns: list[str]) -> list[str]:
    mentions: list[str] = []
    for column in columns:
        candidate = column.strip()
        if not candidate:
            continue
        pattern = re.compile(rf"(?<![\w]){re.escape(candidate)}(?![\w])", re.I)
        if pattern.search(text):
            mentions.append(column)
        if len(mentions) >= 100:
            break
    return mentions


def title_hints(text: str) -> list[str]:
    hints: list[str] = []
    for raw_line in text.splitlines()[:80]:
        line = raw_line.strip().lstrip("#").strip()
        if not line or len(line) > 180:
            continue
        if (
            raw_line.lstrip().startswith("#")
            or re.search(r"dataset|study|survey|数据集|研究|调查", line, re.I)
        ):
            hints.append(line)
        if len(hints) >= 5:
            break
    return list(dict.fromkeys(hints))


def relative_depth(path: Path, root: Path) -> int:
    return max(len(path.relative_to(root).parts) - 1, 0)


def is_ignored(path: Path, root: Path, output_dir: Path) -> bool:
    if path == output_dir or output_dir in path.parents:
        return True
    relative_parts = path.relative_to(root).parts
    return any(part in IGNORED_DIRECTORY_NAMES or part.startswith(".") for part in relative_parts[:-1])


def discover_files(root: Path, table: Path, output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path == table:
            continue
        if path.name.casefold() in IGNORED_FILE_NAMES:
            continue
        if relative_depth(path, root) > MAX_DEPTH:
            continue
        if is_ignored(path, root, output_dir):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            candidates.append(path)
    return candidates


def file_record(path: Path, root: Path, table: Path, columns: list[str]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    category, name_score = category_for(path)
    excerpt = ""
    if suffix in TEXT_SUFFIXES:
        excerpt, extraction_status = extract_text_file(path)
    elif suffix == ".docx":
        excerpt, extraction_status = extract_docx(path)
    elif suffix in TABULAR_SUFFIXES and (name_score > 0 or path.stem.casefold() == table.stem.casefold()):
        excerpt, extraction_status = extract_tabular_context(path)
    elif suffix in TABULAR_SUFFIXES:
        extraction_status = "not_extracted_low_relevance"
    else:
        extraction_status = "requires_parse_document"
    excerpt = normalized_excerpt(excerpt)
    mentions = column_mentions(excerpt, columns) if excerpt else []
    same_stem = path.stem.casefold() == table.stem.casefold()
    score = name_score + min(len(mentions), 5) + (3 if same_stem else 0)
    if excerpt:
        score += 1
    return {
        "path": path.relative_to(root).as_posix(),
        "suffix": suffix,
        "category": category,
        "relevance_score": score,
        "size_bytes": path.stat().st_size,
        "extraction_status": extraction_status,
        "requires_parse_document": extraction_status == "requires_parse_document",
        "column_mentions": mentions,
        "title_hints": title_hints(excerpt) if excerpt else [],
        "excerpt": excerpt,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 数据集背景文件发现结果",
        "",
        f"- 主数据表：`{payload['source_table']}`",
        f"- 扫描目录：`{payload['context_root']}`",
        f"- 发现候选说明文件：{len(payload['companion_files'])}",
        "- 说明：候选文件仅作为数据集背景证据，不能单独证明数据集身份或字段含义。",
        "",
        "| 文件 | 类型 | 类别 | 相关度 | 提取状态 | 命中字段 |",
        "|---|---|---|---:|---|---|",
    ]
    for item in payload["companion_files"]:
        mentions = "、".join(item["column_mentions"][:8]) or "—"
        lines.append(
            f"| `{item['path']}` | `{item['suffix']}` | {item['category']} | "
            f"{item['relevance_score']} | {item['extraction_status']} | {mentions} |"
        )
    if not payload["companion_files"]:
        lines.append("| — | — | — | 0 | 未发现 | — |")
    lines.extend(
        [
            "",
            "## 后续处理",
            "",
            "- `extracted`：可读取 `dataset-context.json` 中的有限长度摘录。",
            "- `requires_parse_document`：由 Agent 使用本地 `parse_document` 工具解析。",
            "- 所有字段含义仍需结合来源、版本、实际列名和用户确认。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    table = args.table.resolve()
    profile_path = args.profile.resolve()
    output_dir = args.output_dir.resolve()
    root = (args.context_root or table.parent).resolve()

    if not table.is_file():
        print(f"Table not found: {table}", file=sys.stderr)
        return 2
    if not profile_path.is_file():
        print(f"Profile not found: {profile_path}", file=sys.stderr)
        return 2
    if root != table.parent and root not in table.parents:
        print("Context root must contain the primary table.", file=sys.stderr)
        return 2

    try:
        profile = read_profile(profile_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    columns = profile_columns(profile)
    records = [
        file_record(path, root, table, columns)
        for path in discover_files(root, table, output_dir)
    ]
    records.sort(key=lambda item: (-item["relevance_score"], item["path"].casefold()))
    records = records[:MAX_FILES]

    payload = {
        "source_table": table.name,
        "context_root": str(root),
        "generated_at": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "network_used": False,
        "identity_confirmed": False,
        "profile_columns": columns,
        "embedded_context_sheets": profile.get("context_sheet_candidates", []),
        "companion_files": records,
        "unparsed_relevant_files": [
            item["path"]
            for item in records
            if item["requires_parse_document"] and item["relevance_score"] > 0
        ],
        "warnings": [
            "File names, excerpts, and matching column names are evidence candidates, not proof of dataset identity.",
            "Dataset source, version, unit of analysis, coding direction, and field meanings still require verification.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset-context.json"
    md_path = output_dir / "dataset-context.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "source_table": table.name,
                "companion_file_count": len(records),
                "unparsed_relevant_files": payload["unparsed_relevant_files"],
                "outputs": [str(json_path), str(md_path)],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
