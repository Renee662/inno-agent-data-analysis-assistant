#!/usr/bin/env python3
"""Smoke-test local dataset context discovery with no network access."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def write_minimal_docx(path: Path, text: str) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document_xml)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    skill_root = app_root / "presets" / "data-analysis-assistant" / ".skills" / "tabular-data-profiler"
    profiler = skill_root / "scripts" / "profile_data.py"
    discovery = skill_root / "scripts" / "discover_context.py"

    with tempfile.TemporaryDirectory(prefix="inno-dataset-context-") as temp_dir:
        fixture = Path(temp_dir)
        table = fixture / "speed_dating.xlsx"
        output_dir = fixture / "generated"
        data = pd.DataFrame(
            {
                "iid": [1, 1, 2],
                "pid": [2, 3, 1],
                "match": [1, 0, 1],
            }
        )
        codebook_sheet = pd.DataFrame(
            {
                "variable": ["iid", "pid", "match"],
                "meaning": ["participant id", "partner id", "mutual match"],
                "unit": ["identifier", "identifier", "0/1"],
            }
        )
        with pd.ExcelWriter(table) as writer:
            data.to_excel(writer, sheet_name="Data", index=False)
            codebook_sheet.to_excel(writer, sheet_name="Codebook", index=False)

        (fixture / "README.md").write_text(
            "# Speed Dating study\nEach row records one participant rating a partner. "
            "The match field indicates a mutual match.",
            encoding="utf-8",
        )
        (fixture / "speed_dating_codebook.csv").write_text(
            "variable,description\niid,participant identifier\npid,partner identifier\nmatch,mutual match\n",
            encoding="utf-8",
        )
        (fixture / "unrelated.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        (fixture / "agent.md").write_text("Workspace instructions", encoding="utf-8")
        (fixture / "analysis-task.json").write_text('{"status":"draft"}', encoding="utf-8")
        write_minimal_docx(
            fixture / "questionnaire.docx",
            "Survey instrument: iid identifies the participant and match is the outcome.",
        )
        (fixture / "study_overview.pdf").write_bytes(b"%PDF-1.4 smoke fixture")

        run(
            [
                sys.executable,
                str(profiler),
                str(table),
                "--output-dir",
                str(output_dir),
            ]
        )
        run(
            [
                sys.executable,
                str(discovery),
                "--table",
                str(table),
                "--profile",
                str(output_dir / "data-profile.json"),
                "--context-root",
                str(fixture),
                "--output-dir",
                str(output_dir),
            ]
        )

        profile = json.loads((output_dir / "data-profile.json").read_text(encoding="utf-8"))
        context = json.loads((output_dir / "dataset-context.json").read_text(encoding="utf-8"))
        context_sheets = {item["sheet"] for item in profile["context_sheet_candidates"]}
        assert context_sheets == {"Codebook"}, context_sheets

        records = {item["path"]: item for item in context["companion_files"]}
        assert "agent.md" not in records
        assert "analysis-task.json" not in records
        assert records["README.md"]["extraction_status"] == "extracted"
        assert "match" in records["README.md"]["column_mentions"]
        assert records["questionnaire.docx"]["extraction_status"] == "extracted"
        assert "iid" in records["questionnaire.docx"]["column_mentions"]
        assert records["speed_dating_codebook.csv"]["extraction_status"] == "extracted"
        assert records["unrelated.csv"]["extraction_status"] == "not_extracted_low_relevance"
        assert records["study_overview.pdf"]["requires_parse_document"] is True
        assert "study_overview.pdf" in context["unparsed_relevant_files"]
        assert context["network_used"] is False

        print(
            json.dumps(
                {
                    "ok": True,
                    "context_sheets": sorted(context_sheets),
                    "companion_files": sorted(records),
                    "network_used": context["network_used"],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
