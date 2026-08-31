#!/usr/bin/env python3
"""Look up public dataset metadata and compare its fields with a local profile."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_OPENML_APIS = (
    "https://www.openml.org/api/v1/json",
    "https://www.openml.org/api_new/v1/json",
)
MAX_CANDIDATES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--claimed-version")
    parser.add_argument("--sheet")
    parser.add_argument(
        "--source-record",
        type=Path,
        help="Local JSON source record; skips online catalog lookup",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--api-base", help=argparse.SUPPRESS)
    return parser.parse_args()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def profile_columns(profile: dict[str, Any], requested_sheet: str | None) -> tuple[str, list[str]]:
    sheets = profile.get("profiles")
    if not isinstance(sheets, list) or not sheets:
        raise ValueError("Profile JSON contains no profiled sheets")
    if requested_sheet:
        selected = next(
            (item for item in sheets if isinstance(item, dict) and item.get("sheet") == requested_sheet),
            None,
        )
        if selected is None:
            available = [item.get("sheet") for item in sheets if isinstance(item, dict)]
            raise ValueError(f"Sheet {requested_sheet!r} not found. Available sheets: {available}")
    elif len(sheets) == 1:
        selected = sheets[0]
    else:
        available = [item.get("sheet") for item in sheets if isinstance(item, dict)]
        raise ValueError(f"Multiple profiled sheets require --sheet. Available sheets: {available}")
    if not isinstance(selected, dict):
        raise ValueError("Selected profile sheet is invalid")
    columns = [
        str(item["column"])
        for item in selected.get("columns", [])
        if isinstance(item, dict) and isinstance(item.get("column"), str)
    ]
    if not columns:
        raise ValueError("Selected profile sheet contains no columns")
    return str(selected.get("sheet", "")), columns


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Inno-Agent-Public-Dataset-Verifier/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed catalog URL
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Public catalog request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Public catalog returned a non-object response for {url}")
    return payload


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def search_name_variants(name: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", name.strip())
    variants = [
        cleaned,
        cleaned.replace(" ", "-"),
        cleaned.replace(" ", "_"),
        cleaned.replace(" ", ""),
    ]
    return list(dict.fromkeys(value for value in variants if value))


def validate_api_base(api_base: str) -> str:
    parsed = urlparse(api_base)
    allowed_openml = parsed.scheme == "https" and parsed.hostname == "www.openml.org"
    allowed_test_server = (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
    )
    if not (allowed_openml or allowed_test_server):
        raise ValueError("Catalog API base must be official OpenML HTTPS")
    return api_base.rstrip("/")


def openml_records(
    name: str, api_base: str, timeout: float
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    api_base = validate_api_base(api_base)
    records: dict[str, dict[str, Any]] = {}
    queried_urls: list[str] = []
    lookup_errors: list[str] = []
    for variant in search_name_variants(name):
        url = f"{api_base.rstrip('/')}/data/list/data_name/{quote(variant, safe='')}/limit/20"
        queried_urls.append(url)
        try:
            payload = fetch_json(url, timeout)
        except ValueError as exc:
            lookup_errors.append(str(exc))
            continue
        datasets = payload.get("data", {}).get("dataset", [])
        if isinstance(datasets, dict):
            datasets = [datasets]
        for item in datasets if isinstance(datasets, list) else []:
            if not isinstance(item, dict):
                continue
            dataset_id = str(item.get("did") or item.get("id") or "").strip()
            if dataset_id:
                records[dataset_id] = item

    ranked = sorted(
        records.items(),
        key=lambda pair: (
            -difflib.SequenceMatcher(
                None,
                normalized_name(name),
                normalized_name(str(pair[1].get("name", ""))),
            ).ratio(),
            int(pair[0]) if pair[0].isdigit() else sys.maxsize,
        ),
    )[:MAX_CANDIDATES]

    candidates: list[dict[str, Any]] = []
    for dataset_id, summary in ranked:
        detail_url = f"{api_base.rstrip('/')}/data/{quote(dataset_id, safe='')}"
        features_url = f"{api_base.rstrip('/')}/data/features/{quote(dataset_id, safe='')}"
        queried_urls.extend([detail_url, features_url])
        try:
            detail_payload = fetch_json(detail_url, timeout)
            features_payload = fetch_json(features_url, timeout)
        except ValueError as exc:
            lookup_errors.append(str(exc))
            candidates.append(
                {
                    "dataset_id": dataset_id,
                    "name": summary.get("name"),
                    "version": summary.get("version"),
                    "status": summary.get("status"),
                    "lookup_error": str(exc),
                    "columns": [],
                }
            )
            continue
        detail = detail_payload.get("data_set_description", {})
        if not isinstance(detail, dict):
            detail = {}
        features = features_payload.get("data_features", {}).get("feature", [])
        if isinstance(features, dict):
            features = [features]
        columns = [
            str(item["name"])
            for item in features if isinstance(item, dict) and item.get("name") is not None
        ]
        field_definitions = {
            str(item["name"]): {
                "data_type": item.get("data_type"),
                "nominal_values": item.get("nominal_value"),
                "is_target": item.get("is_target"),
                "is_ignore": item.get("is_ignore"),
                "is_row_identifier": item.get("is_row_identifier"),
            }
            for item in features
            if isinstance(item, dict) and item.get("name") is not None
        }
        candidates.append(
            {
                "catalog": "OpenML",
                "dataset_id": dataset_id,
                "name": detail.get("name") or summary.get("name"),
                "version": str(detail.get("version") or summary.get("version") or ""),
                "version_label": detail.get("version_label"),
                "status": detail.get("status") or summary.get("status"),
                "publisher": "OpenML",
                "creator": detail.get("creator"),
                "description": detail.get("description"),
                "citation": detail.get("citation"),
                "licence": detail.get("licence"),
                "original_data_url": detail.get("original_data_url"),
                "source_title": f"OpenML dataset {dataset_id}",
                "source_url": f"https://www.openml.org/d/{dataset_id}",
                "metadata_url": detail_url,
                "features_url": features_url,
                "columns": columns,
                "field_definitions": field_definitions,
            }
        )
    return candidates, queried_urls, lookup_errors


def openml_catalog_lookup(
    name: str, api_base: str | None, timeout: float
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    queried_urls: list[str] = []
    lookup_errors: list[str] = []
    bases = (api_base,) if api_base else DEFAULT_OPENML_APIS
    for base in bases:
        candidates, current_urls, current_errors = openml_records(name, base, timeout)
        queried_urls.extend(current_urls)
        lookup_errors.extend(current_errors)
        if candidates:
            return candidates, queried_urls, lookup_errors
    return [], queried_urls, lookup_errors


def local_source_record(path: Path) -> dict[str, Any]:
    record = read_json(path, "source record")
    required = ("dataset_name", "source_url", "source_title", "publisher", "version", "columns")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("Source record is missing required fields: " + ", ".join(missing))
    if not isinstance(record["columns"], list) or not all(
        isinstance(value, str) and value.strip() for value in record["columns"]
    ):
        raise ValueError("Source record columns must be a non-empty string array")
    source_url = urlparse(str(record["source_url"]))
    if source_url.scheme not in {"http", "https"} or not source_url.hostname:
        raise ValueError("Source record source_url must be an HTTP(S) URL")
    return {
        "catalog": record.get("catalog", "other-public-source"),
        "dataset_id": record.get("dataset_id"),
        "name": record["dataset_name"],
        "version": str(record["version"]),
        "version_label": record.get("version_label"),
        "status": record.get("status"),
        "publisher": record["publisher"],
        "creator": record.get("creator"),
        "description": record.get("description"),
        "citation": record.get("citation"),
        "licence": record.get("licence"),
        "original_data_url": record.get("original_data_url"),
        "source_title": record["source_title"],
        "source_url": record["source_url"],
        "metadata_url": record.get("metadata_url", record["source_url"]),
        "features_url": record.get("features_url", record["source_url"]),
        "columns": list(dict.fromkeys(value.strip() for value in record["columns"])),
        "field_definitions": (
            record.get("field_definitions")
            if isinstance(record.get("field_definitions"), dict)
            else {}
        ),
    }


def compare_columns(
    observed: list[str],
    reference: list[str],
    claimed_name: str,
    candidate_name: str,
    claimed_version: str | None,
    candidate_version: str,
) -> dict[str, Any]:
    observed_map = {value.casefold(): value for value in observed}
    reference_map = {value.casefold(): value for value in reference}
    shared_keys = observed_map.keys() & reference_map.keys()
    matched = [observed_map[key] for key in observed_map if key in shared_keys]
    missing = [reference_map[key] for key in reference_map if key not in observed_map]
    extra = [observed_map[key] for key in observed_map if key not in reference_map]
    case_differences = [
        {"uploaded": observed_map[key], "reference": reference_map[key]}
        for key in shared_keys
        if observed_map[key] != reference_map[key]
    ]
    reference_coverage = len(matched) / max(len(reference), 1)
    uploaded_coverage = len(matched) / max(len(observed), 1)
    exact_columns = not missing and not extra
    name_similarity = difflib.SequenceMatcher(
        None, normalized_name(claimed_name), normalized_name(candidate_name)
    ).ratio()
    version_match = (
        None
        if not claimed_version
        else claimed_version.strip().casefold() == candidate_version.strip().casefold()
    )

    if exact_columns and name_similarity >= 0.9 and version_match is not False:
        status = "strong-candidate"
    elif reference_coverage >= 0.8 and uploaded_coverage >= 0.8 and name_similarity >= 0.7:
        status = "partial-column-match"
    else:
        status = "insufficient-column-match"
    return {
        "candidate_status": status,
        "identity_confirmed": False,
        "name_similarity": round(name_similarity, 4),
        "claimed_version": claimed_version,
        "version_match": version_match,
        "exact_columns": exact_columns,
        "matched_columns": matched,
        "missing_from_upload": missing,
        "extra_in_upload": extra,
        "case_differences": case_differences,
        "reference_column_coverage": round(reference_coverage, 4),
        "uploaded_column_coverage": round(uploaded_coverage, 4),
        "requires_user_confirmation": True,
        "unresolved_reasons": [
            reason
            for reason, active in (
                ("用户未提供版本，需核对上传文件对应的版本", not claimed_version),
                ("用户提供的版本与公开来源版本不一致", version_match is False),
                ("上传文件与公开代码本列名并非完全一致", not exact_columns),
                ("数据集名称与公开来源名称不完全一致", name_similarity < 0.9),
            )
            if active
        ],
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 公开数据集来源核验",
        "",
        f"- 用户提供名称：`{payload['claimed_dataset_name']}`",
        f"- 用户提供版本：`{payload['claimed_version'] or '未提供'}`",
        f"- 实际工作表：`{payload['sheet']}`",
        f"- 实际列数：{len(payload['observed_columns'])}",
        f"- 是否联网：{'是' if payload['network_used'] else '否'}",
        "- 结论边界：所有结果均为候选核验，用户确认前 `identity_confirmed` 始终为 `false`。",
        "",
        "| 候选来源 | 版本 | 状态 | 匹配列 | 代码本覆盖 | 上传列覆盖 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in payload["candidates"]:
        comparison = item["comparison"]
        lines.append(
            f"| [{item.get('name') or '未命名'}]({item.get('source_url')}) | "
            f"{item.get('version') or '未知'} | {comparison['candidate_status']} | "
            f"{len(comparison['matched_columns'])} | "
            f"{comparison['reference_column_coverage']:.1%} | "
            f"{comparison['uploaded_column_coverage']:.1%} |"
        )
        lines.append(
            f"| ↳ 列差异 |  |  | 缺失：{', '.join(comparison['missing_from_upload'][:12]) or '无'} | "
            f"额外：{', '.join(comparison['extra_in_upload'][:12]) or '无'} |  |"
        )
    if not payload["candidates"]:
        lines.append("| 未找到 | — | lookup-no-result | 0 | 0% | 0% |")
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "- 核对来源页面、数据集 ID、版本/年份和列名差异。",
            "- 只把匹配的公开定义作为候选字段含义；缺失、额外或冲突字段继续询问用户。",
            "- 不得仅因数据集名称或部分列相似就认定身份。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        profile = read_json(args.profile.resolve(), "profile")
        sheet, observed = profile_columns(profile, args.sheet)
        if args.source_record:
            candidates = [local_source_record(args.source_record.resolve())]
            queried_urls: list[str] = []
            lookup_errors: list[str] = []
            network_used = False
            lookup_mode = "provided-source-record"
        else:
            candidates, queried_urls, lookup_errors = openml_catalog_lookup(
                args.dataset_name, args.api_base, args.timeout
            )
            network_used = True
            lookup_mode = "openml-metadata-only"

        compared = []
        for candidate in candidates:
            comparison = compare_columns(
                observed,
                candidate.get("columns", []),
                args.dataset_name,
                str(candidate.get("name", "")),
                args.claimed_version,
                str(candidate.get("version", "")),
            )
            compared.append({**candidate, "comparison": comparison})
        compared.sort(
            key=lambda item: (
                {"strong-candidate": 0, "partial-column-match": 1}.get(
                    item["comparison"]["candidate_status"], 2
                ),
                -item["comparison"]["name_similarity"],
                -len(item["comparison"]["matched_columns"]),
            )
        )

        payload = {
            "claimed_dataset_name": args.dataset_name,
            "claimed_version": args.claimed_version,
            "source_profile": str(args.profile),
            "sheet": sheet,
            "observed_columns": observed,
            "lookup_mode": lookup_mode,
            "network_used": network_used,
            "external_data_sent": (
                {"dataset_name": args.dataset_name, "uploaded_table_data": False, "uploaded_columns": False}
                if network_used
                else {}
            ),
            "catalog_queries": queried_urls,
            "catalog_errors": lookup_errors,
            "lookup_status": (
                "catalog-unavailable"
                if not candidates and lookup_errors
                else "no-result"
                if not candidates
                else "completed-with-errors"
                if lookup_errors
                else "completed"
            ),
            "retrieved_at": datetime.now().astimezone().isoformat(),
            "identity_confirmed": False,
            "candidates": compared,
            "warnings": [
                "A public dataset name or similar columns do not prove dataset identity.",
                "Version and actual uploaded columns must be checked before using codebook meanings.",
                "Only the public dataset name is sent to OpenML; uploaded data and column names remain local.",
            ],
        }
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "public-dataset-verification.json"
        md_path = output_dir / "public-dataset-verification.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(markdown_report(payload), encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "candidate_count": len(compared),
                    "best_status": (
                        compared[0]["comparison"]["candidate_status"]
                        if compared
                        else "lookup-no-result"
                    ),
                    "identity_confirmed": False,
                    "outputs": [str(md_path), str(json_path)],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except ValueError as exc:
        print(f"Public dataset verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
