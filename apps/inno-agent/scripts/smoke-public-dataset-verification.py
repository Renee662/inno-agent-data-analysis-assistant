#!/usr/bin/env python3
"""Smoke-test public catalog lookup and local column/version verification."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


class OpenMLFixtureHandler(BaseHTTPRequestHandler):
    requests: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self.__class__.requests.append(self.path)
        decoded = unquote(self.path)
        if "/data/list/data_name/" in decoded:
            if decoded.endswith("/SpeedDating/limit/20"):
                payload = {
                    "data": {
                        "dataset": [
                            {
                                "did": "40536",
                                "name": "SpeedDating",
                                "version": "1",
                                "status": "active",
                            }
                        ]
                    }
                }
            else:
                payload = {"data": {"dataset": []}}
        elif decoded.endswith("/data/40536"):
            payload = {
                "data_set_description": {
                    "id": "40536",
                    "name": "SpeedDating",
                    "version": "1",
                    "status": "active",
                    "creator": "fixture creator",
                    "description": "Fixture metadata only",
                    "licence": "public fixture",
                }
            }
        elif decoded.endswith("/data/features/40536"):
            payload = {
                "data_features": {
                    "feature": [
                        {"index": str(index), "name": name, "data_type": "numeric"}
                        for index, name in enumerate(["iid", "pid", "match", "attr", "sinc"])
                    ]
                }
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def write_profile(path: Path) -> None:
    columns = ["iid", "pid", "match", "attr", "sinc"]
    source_sha256 = "0" * 64
    payload = {
        "source_file": "speed_dating.csv",
        "table_structure_status": "confirmed",
        "table_read_spec": {
            "schema_version": 1,
            "source_file": "speed_dating.csv",
            "source_sha256": source_sha256,
            "read_only": True,
            "sheets": [
                {
                    "sheet": "speed_dating",
                    "status": "auto-confirmed",
                    "requires_user_confirmation": False,
                    "selection_source": "detector",
                    "confidence": "high",
                    "recommended_candidate_id": "header-row-1",
                    "selected": {
                        "id": "header-row-1",
                        "sheet": "speed_dating",
                        "source_suffix": ".csv",
                        "source_sha256": source_sha256,
                        "header_mode": "single-row",
                        "header_rows": [1],
                        "data_start_row": 2,
                        "column_names": columns,
                    },
                    "candidates": [],
                }
            ],
        },
        "profiles": [
            {
                "sheet": "speed_dating",
                "row_count": 100,
                "column_count": len(columns),
                "columns": [{"column": name} for name in columns],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    verifier = (
        app_root
        / "presets"
        / "data-analysis-assistant"
        / ".skills"
        / "tabular-data-profiler"
        / "scripts"
        / "verify_public_dataset.py"
    )
    planner = (
        app_root
        / "presets"
        / "data-analysis-assistant"
        / ".skills"
        / "plan-relationship-analysis"
        / "scripts"
        / "build_analysis_plan.py"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenMLFixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="inno-public-dataset-") as temp_dir:
            fixture = Path(temp_dir)
            profile = fixture / "data-profile.json"
            output_dir = fixture / "output"
            write_profile(profile)
            api_base = f"http://127.0.0.1:{server.server_port}/api/v1/json"
            run(
                [
                    sys.executable,
                    str(verifier),
                    "--profile",
                    str(profile),
                    "--dataset-name",
                    "Speed Dating",
                    "--api-base",
                    api_base,
                    "--output-dir",
                    str(output_dir),
                ]
            )
            payload = json.loads(
                (output_dir / "public-dataset-verification.json").read_text(
                    encoding="utf-8"
                )
            )
            assert payload["lookup_status"] == "completed"
            assert payload["network_used"] is True
            assert payload["external_data_sent"]["uploaded_table_data"] is False
            assert payload["external_data_sent"]["uploaded_columns"] is False
            assert payload["identity_confirmed"] is False
            assert len(payload["candidates"]) == 1
            candidate = payload["candidates"][0]
            comparison = candidate["comparison"]
            assert candidate["dataset_id"] == "40536"
            assert candidate["version"] == "1"
            assert comparison["candidate_status"] == "strong-candidate"
            assert comparison["exact_columns"] is True
            assert comparison["version_match"] is None
            assert comparison["identity_confirmed"] is False
            assert any("未提供版本" in item for item in comparison["unresolved_reasons"])
            assert set(comparison["matched_columns"]) == {
                "iid",
                "pid",
                "match",
                "attr",
                "sinc",
            }
            assert all("attr" not in request for request in OpenMLFixtureHandler.requests)
            assert all("sinc" not in request for request in OpenMLFixtureHandler.requests)

            plan_output = fixture / "plan"
            run(
                [
                    sys.executable,
                    str(planner),
                    "--profile",
                    str(profile),
                    "--public-verification",
                    str(output_dir / "public-dataset-verification.json"),
                    "--goal",
                    "association",
                    "--decision-goal",
                    "relationships",
                    "--outcome",
                    "match",
                    "--predictors",
                    "attr",
                    "sinc",
                    "--output-dir",
                    str(plan_output),
                ]
            )
            task = json.loads(
                (plan_output / "analysis-task.json").read_text(encoding="utf-8")
            )
            semantic_fields = {
                item["column"]: item for item in task["semantic_review"]["fields"]
            }
            public_evidence = semantic_fields["match"]["source_evidence"][0]
            assert public_evidence["source_type"] == "public-dataset-candidate"
            assert public_evidence["source_path"] == "https://www.openml.org/d/40536"
            assert public_evidence["version"] == "1"
            assert public_evidence["identity_confirmed"] is False

            source_record = fixture / "public-source-record.json"
            source_record.write_text(
                json.dumps(
                    {
                        "dataset_name": "Speed Dating",
                        "dataset_id": "official-1",
                        "source_title": "Official codebook",
                        "publisher": "Fixture Repository",
                        "source_url": "https://example.org/codebook",
                        "version": "2",
                        "columns": ["iid", "pid", "match", "attr", "sinc"],
                    }
                ),
                encoding="utf-8",
            )
            local_output = fixture / "local-output"
            run(
                [
                    sys.executable,
                    str(verifier),
                    "--profile",
                    str(profile),
                    "--dataset-name",
                    "Speed Dating",
                    "--claimed-version",
                    "1",
                    "--source-record",
                    str(source_record),
                    "--output-dir",
                    str(local_output),
                ]
            )
            local_payload = json.loads(
                (local_output / "public-dataset-verification.json").read_text(
                    encoding="utf-8"
                )
            )
            local_comparison = local_payload["candidates"][0]["comparison"]
            assert local_payload["network_used"] is False
            assert local_comparison["version_match"] is False
            assert local_comparison["candidate_status"] == "partial-column-match"
            assert local_comparison["identity_confirmed"] is False

            print(
                json.dumps(
                    {
                        "ok": True,
                        "lookup_status": payload["lookup_status"],
                        "candidate_status": comparison["candidate_status"],
                        "version_unresolved": comparison["version_match"] is None,
                        "mismatched_version_status": local_comparison["candidate_status"],
                        "identity_confirmed": payload["identity_confirmed"],
                        "planner_public_evidence": public_evidence["source_type"],
                        "uploaded_columns_sent": payload["external_data_sent"][
                            "uploaded_columns"
                        ],
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
