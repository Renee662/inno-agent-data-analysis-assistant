#!/usr/bin/env python3
"""Create a read-only structural and data-quality profile for tabular files."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised when dependencies are absent
    print(
        "Missing dependency. Install packages from "
        "managed shared data-analysis environment before retrying.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from file_utils import sha256_file as file_sha256  # noqa: E402
from table_utils import unique_column_names  # noqa: E402


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}
DEFAULT_MISSING_VALUE_TOKENS = ("?",)
SENSITIVE_NAME_RE = re.compile(
    r"^(name|full.?name|person.?name|user.?name|student.?name|employee.?name|"
    r"customer.?name|patient.?name)$|"
    r"(^|[_\-\s])(phone|mobile|email|address|passport|national.?id|identity|"
    r"student.?id|employee.?id|sex|gender|race|ethnicity|marital.?status|"
    r"nationality|native.?country)([_\-\s]|$)|"
    r"姓名|名字|电话|手机|邮箱|地址|住址|身份证|护照|学号|工号|"
    r"性别|种族|族裔|民族|婚姻|国籍|原籍国",
    re.IGNORECASE,
)
OPAQUE_COLUMN_NAME_RE = re.compile(
    r"^(?:column_?\d+|unnamed(?::\s*\d+)?|x\d+|var(?:iable)?_?\d+|col(?:umn)?_?\d+)$",
    re.IGNORECASE,
)
ID_NAME_RE = re.compile(r"(^|[_\-\s])(id|key|code|number|no)([_\-\s]|$)|编号|编码", re.IGNORECASE)
CONTEXT_SHEET_RE = re.compile(
    r"read.?me|code.?book|data.?dictionary|meta.?data|variable|label|schema|"
    r"说明|介绍|字典|变量|字段|编码|问卷",
    re.IGNORECASE,
)
CONTEXT_COLUMN_RE = re.compile(
    r"variable|field|column|name|label|description|meaning|value|code|unit|"
    r"变量|字段|列名|名称|标签|说明|含义|取值|编码|单位",
    re.IGNORECASE,
)
CONTEXT_FIELD_COLUMN_RE = re.compile(
    r"variable|field|column|var.?name|字段|变量|列名",
    re.IGNORECASE,
)
CONTEXT_DESCRIPTION_COLUMN_RE = re.compile(
    r"label|description|meaning|value.?label|unit|说明|含义|标签|取值|单位",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV, TSV, XLSX, or XLS file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work/data-profile"),
        help="Directory for generated profile artifacts",
    )
    parser.add_argument("--sheet", help="Profile only this Excel sheet")
    parser.add_argument(
        "--header-rows",
        help="Confirmed 1-based header row numbers, for example 2 or 1,2",
    )
    parser.add_argument(
        "--headerless",
        action="store_true",
        help="Confirm that the table has no header row",
    )
    parser.add_argument(
        "--structure-confirmed",
        action="store_true",
        help="Record that the user explicitly confirmed --header-rows or --headerless",
    )
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def is_missing_token(value: Any, tokens: tuple[str, ...] = DEFAULT_MISSING_VALUE_TOKENS) -> bool:
    return isinstance(value, str) and value.strip() in tokens


def normalize_missing_tokens(
    frame: pd.DataFrame, tokens: tuple[str, ...] = DEFAULT_MISSING_VALUE_TOKENS
) -> pd.DataFrame:
    """Convert configured textual sentinels to real missing values without trimming other text."""
    result = frame.copy()
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        mask = series.map(lambda value: is_missing_token(value, tokens))
        if bool(mask.any()):
            result.loc[mask, column] = pd.NA
    return result


def read_delimited(
    path: Path, separator: str | None, header: int | list[int] | None = 0
) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            if separator is None:
                return pd.read_csv(
                    path, encoding=encoding, sep=None, engine="python", header=header
                )
            return pd.read_csv(path, encoding=encoding, sep=separator, header=header)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Unable to decode delimited file. " + " | ".join(errors))


def load_raw_tables(
    path: Path, requested_sheet: str | None
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {"CSV": read_delimited(path, None, header=None)}, ["CSV"]
    if suffix == ".tsv":
        return {"TSV": read_delimited(path, "\t", header=None)}, ["TSV"]

    workbook = pd.ExcelFile(path)
    sheet_names = list(workbook.sheet_names)
    if requested_sheet:
        if requested_sheet not in sheet_names:
            raise ValueError(
                f"Sheet {requested_sheet!r} not found. Available sheets: {sheet_names}"
            )
        selected = [requested_sheet]
    else:
        selected = sheet_names
    return {
        name: pd.read_excel(workbook, sheet_name=name, header=None) for name in selected
    }, sheet_names


PLACEHOLDER_HEADER_RE = re.compile(
    r"^(?:x\d+|y\d*|var(?:iable)?\d+|col(?:umn)?\d+|unnamed(?::\s*\d+)?)$",
    re.IGNORECASE,
)
HEADER_NAME_RE = re.compile(r"[A-Za-z_\-\s\u4e00-\u9fff]")


def parse_header_rows(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        rows = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--header-rows must contain positive 1-based row numbers") from exc
    if not rows or any(row < 1 for row in rows) or len(rows) > 3:
        raise ValueError("--header-rows must contain one to three positive row numbers")
    return rows


def header_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def column_names_for(raw: pd.DataFrame, header_rows: list[int]) -> list[str]:
    if not header_rows:
        return [f"column_{index + 1}" for index in range(raw.shape[1])]
    zero_based = [row - 1 for row in header_rows]
    names: list[str] = []
    for column_index in range(raw.shape[1]):
        parts = [header_text(raw.iat[row, column_index]) for row in zero_based]
        names.append(" | ".join(part for part in parts if part))
    return unique_column_names(names)


def detected_family(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "numeric"
    text = str(value).strip()
    if not text:
        return "missing"
    try:
        float(text.replace(",", ""))
        return "numeric"
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+.*)?", text):
        return "datetime"
    return "text"


def data_consistency(raw: pd.DataFrame, data_start_row: int) -> tuple[float, int, list[str]]:
    sample = raw.iloc[data_start_row - 1 : data_start_row - 1 + 200]
    consistencies: list[float] = []
    dominant_families: list[str] = []
    mixed_columns = 0
    for column_index in range(raw.shape[1]):
        families = [
            detected_family(value)
            for value in sample.iloc[:, column_index].tolist()
            if detected_family(value) != "missing"
        ]
        if not families:
            dominant_families.append("missing")
            continue
        counts = {family: families.count(family) for family in set(families)}
        dominant = max(counts, key=counts.get)
        dominant_families.append(dominant)
        consistencies.append(counts[dominant] / len(families))
        if len(counts) > 1:
            mixed_columns += 1
    return (
        sum(consistencies) / len(consistencies) if consistencies else 0.0,
        mixed_columns,
        dominant_families,
    )


def candidate_for(raw: pd.DataFrame, header_rows: list[int], candidate_id: str) -> dict[str, Any]:
    data_start_row = max(header_rows) + 1 if header_rows else 1
    names = column_names_for(raw, header_rows)
    consistency, mixed_columns, dominant = data_consistency(raw, data_start_row)
    if header_rows:
        cells = [
            raw.iat[row - 1, column]
            for row in header_rows
            for column in range(raw.shape[1])
        ]
        nonempty = [value for value in cells if header_text(value)]
        text_ratio = (
            sum(detected_family(value) == "text" for value in nonempty) / len(nonempty)
            if nonempty
            else 0.0
        )
        name_ratio = sum(bool(HEADER_NAME_RE.search(name)) for name in names) / max(len(names), 1)
        unique_ratio = len(set(names)) / max(len(names), 1)
        placeholder_ratio = sum(bool(PLACEHOLDER_HEADER_RE.fullmatch(name)) for name in names) / max(
            len(names), 1
        )
        last_header = header_rows[-1] - 1
        contrast_values: list[float] = []
        for column, family in enumerate(dominant):
            if family == "missing":
                continue
            contrast_values.append(
                1.0 if detected_family(raw.iat[last_header, column]) != family else 0.0
            )
        contrast = sum(contrast_values) / len(contrast_values) if contrast_values else 0.0
        score = (
            0.22 * text_ratio
            + 0.18 * name_ratio
            + 0.15 * unique_ratio
            + 0.28 * consistency
            + 0.17 * contrast
            - 0.36 * placeholder_ratio
            - 0.04 * max(header_rows[0] - 1, 0)
            - (0.07 if len(header_rows) > 1 else 0.0)
        )
    else:
        first_row = raw.iloc[0] if len(raw) else pd.Series(dtype=object)
        compatible = []
        for column, family in enumerate(dominant):
            if family != "missing" and column < len(first_row):
                compatible.append(1.0 if detected_family(first_row.iloc[column]) == family else 0.0)
        compatibility = sum(compatible) / len(compatible) if compatible else 0.0
        score = 0.25 * consistency + 0.55 * compatibility
        text_ratio = 0.0
        placeholder_ratio = 0.0
        contrast = 0.0
    score = max(0.0, min(1.0, score))
    return {
        "id": candidate_id,
        "header_rows": header_rows,
        "data_start_row": data_start_row,
        "header_mode": "headerless" if not header_rows else ("multi-row" if len(header_rows) > 1 else "single-row"),
        "score": round(score, 4),
        "column_names": names,
        "column_names_preview": names[:12],
        "data_type_consistency": round(consistency, 4),
        "sample_mixed_type_column_count": mixed_columns,
        "placeholder_name_count": sum(bool(PLACEHOLDER_HEADER_RE.fullmatch(name)) for name in names),
        "evidence": {
            "header_text_ratio": round(text_ratio, 4),
            "header_data_type_contrast": round(contrast, 4),
        },
    }


def detect_candidates(raw: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    max_header_row = min(3, max(len(raw) - 1, 0))
    for row in range(1, max_header_row + 1):
        candidates.append(candidate_for(raw, [row], f"header-row-{row}"))
    if max_header_row >= 2:
        candidates.append(candidate_for(raw, [1, 2], "header-rows-1-2"))
    if len(raw):
        candidates.append(candidate_for(raw, [], "headerless"))
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def apply_candidate(raw: pd.DataFrame, candidate: dict[str, Any]) -> pd.DataFrame:
    data_start = int(candidate["data_start_row"]) - 1
    frame = raw.iloc[data_start:].copy()
    frame.columns = list(candidate["column_names"])
    frame = normalize_missing_tokens(frame)
    for column in frame.columns:
        series = frame[column]
        non_null_count = int(series.notna().sum())
        if non_null_count == 0 or pd.api.types.is_numeric_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) == non_null_count:
            frame[column] = numeric
            continue
        text_values = series.dropna().astype(str).str.strip()
        if not text_values.empty and text_values.str.fullmatch(
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+.*)?"
        ).all():
            frame[column] = pd.to_datetime(series, errors="coerce")
    return frame


def build_structure_spec(
    source: Path,
    raw_tables: dict[str, pd.DataFrame],
    explicit_header_rows: list[int] | None,
    headerless: bool,
    structure_confirmed: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if explicit_header_rows is not None and headerless:
        raise ValueError("Use either --header-rows or --headerless, not both")
    explicit = explicit_header_rows is not None or headerless
    if explicit != structure_confirmed:
        raise ValueError(
            "An explicit --header-rows/--headerless selection and --structure-confirmed are required together"
        )
    if explicit and len(raw_tables) != 1:
        raise ValueError("Confirm one sheet at a time with --sheet when multiple sheets are profiled")

    source_hash = file_sha256(source)
    tables: dict[str, pd.DataFrame] = {}
    sheet_specs: list[dict[str, Any]] = []
    for sheet_name, raw in raw_tables.items():
        candidates = detect_candidates(raw)
        if not candidates:
            raise ValueError(f"Sheet {sheet_name!r} is empty; table structure cannot be detected")
        if explicit:
            rows = [] if headerless else list(explicit_header_rows or [])
            selected = next(
                (item for item in candidates if item["header_rows"] == rows),
                candidate_for(raw, rows, "user-specified"),
            )
            status = "user-confirmed"
            requires_confirmation = False
            selection_source = "user"
        else:
            selected = candidates[0]
            margin = selected["score"] - (candidates[1]["score"] if len(candidates) > 1 else 0.0)
            standard = selected["header_rows"] == [1]
            requires_confirmation = not standard or selected["score"] < 0.65 or margin < 0.10
            status = "pending-user-confirmation" if requires_confirmation else "auto-confirmed"
            selection_source = "detector"

        margin = selected["score"] - (candidates[1]["score"] if len(candidates) > 1 else 0.0)
        confidence = "high" if selected["score"] >= 0.75 and margin >= 0.15 else (
            "medium" if selected["score"] >= 0.60 and margin >= 0.07 else "low"
        )
        selected_payload = {
            key: selected[key]
            for key in ("id", "header_rows", "data_start_row", "header_mode", "column_names")
        }
        selected_payload.update(
            {
                "sheet": sheet_name,
                "source_suffix": source.suffix.lower(),
                "source_sha256": source_hash,
                "missing_value_tokens": list(DEFAULT_MISSING_VALUE_TOKENS),
            }
        )
        tables[sheet_name] = apply_candidate(raw, selected)
        sheet_specs.append(
            {
                "sheet": sheet_name,
                "status": status,
                "requires_user_confirmation": requires_confirmation,
                "selection_source": selection_source,
                "confidence": confidence,
                "recommended_candidate_id": candidates[0]["id"],
                "selected": selected_payload,
                "candidates": [
                    {key: value for key, value in candidate.items() if key != "column_names"}
                    for candidate in candidates[:4]
                ],
            }
        )
    return tables, {
        "schema_version": 1,
        "source_file": source.name,
        "source_sha256": source_hash,
        "read_only": True,
        "sheets": sheet_specs,
    }


def type_family(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "numeric"
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "datetime"
    return "text"


def sample_values(series: pd.Series, sensitive: bool) -> list[Any]:
    if sensitive:
        return ["[REDACTED]"] if series.notna().any() else []
    values = series.dropna().drop_duplicates().head(3).tolist()
    return [scalar(value) for value in values]


def infer_semantic_type(series: pd.Series, column_name: str, sensitive: bool) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "all-missing"
    if sensitive:
        return "sensitive-candidate"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = unique_count / max(len(non_null), 1)
    if ID_NAME_RE.search(column_name) or (unique_ratio >= 0.98 and unique_count >= 20):
        return "identifier-candidate"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric-discrete" if unique_count <= 20 else "numeric-continuous"
    if unique_count <= 100 and unique_ratio <= 0.2:
        return "categorical"
    return "text"


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())
    else:
        lower, upper, outlier_count = q1, q3, 0
    return {
        "min": float(values.min()),
        "q1": q1,
        "median": float(values.median()),
        "q3": q3,
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": None if len(values) < 2 else float(values.std()),
        "iqr_lower_bound": lower,
        "iqr_upper_bound": upper,
        "iqr_outlier_count": outlier_count,
    }


def profile_column(sheet: str, name: str, series: pd.Series, row_count: int) -> dict[str, Any]:
    non_null = series.dropna()
    missing_count = int(series.isna().sum())
    unique_count = int(non_null.nunique(dropna=True)) if not non_null.empty else 0
    frequencies = non_null.value_counts(dropna=True, normalize=True)
    top_share = float(frequencies.iloc[0]) if not frequencies.empty else None
    sampled_families = sorted(
        {type_family(value) for value in non_null.head(5000).tolist()} - {"missing"}
    )
    sensitive = bool(SENSITIVE_NAME_RE.search(str(name)))
    opaque_name = bool(OPAQUE_COLUMN_NAME_RE.fullmatch(str(name).strip()))
    result: dict[str, Any] = {
        "sheet": sheet,
        "column": str(name),
        "pandas_dtype": str(series.dtype),
        "inferred_type": infer_semantic_type(series, str(name), sensitive),
        "non_null_count": int(series.notna().sum()),
        "missing_count": missing_count,
        "missing_rate": missing_count / max(row_count, 1),
        "unique_count": unique_count,
        "unique_rate_among_non_null": unique_count / max(len(non_null), 1),
        "constant": unique_count <= 1,
        "near_constant": bool(top_share is not None and top_share >= 0.95 and unique_count > 1),
        "most_common_share": top_share,
        "mixed_python_types": len(sampled_families) > 1,
        "observed_type_families": sampled_families,
        "sensitive_name_candidate": sensitive,
        "sensitive_review_status": (
            "name-candidate"
            if sensitive
            else "pending-semantic-review"
            if opaque_name
            else "name-screen-no-match"
        ),
        "sample_values": sample_values(series, sensitive),
    }
    if pd.api.types.is_numeric_dtype(series):
        result["numeric_summary"] = numeric_summary(series)
    return result


def profile_sheet(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy(deep=False)
    frame.columns = [str(column) for column in frame.columns]
    rows, columns = frame.shape
    column_profiles = [
        profile_column(name, column, frame[column], rows) for column in frame.columns
    ]
    sensitive_candidates = [
        item["column"] for item in column_profiles if item["sensitive_name_candidate"]
    ]
    opaque_columns = [
        item["column"]
        for item in column_profiles
        if item["sensitive_review_status"] == "pending-semantic-review"
    ]
    return {
        "sheet": name,
        "row_count": int(rows),
        "column_count": int(columns),
        "duplicate_row_count": int(frame.duplicated().sum()) if rows else 0,
        "fully_empty_row_count": int(frame.isna().all(axis=1).sum()) if rows else 0,
        "fully_empty_column_count": int(frame.isna().all(axis=0).sum()) if columns else 0,
        "sensitive_review": {
            "status": "pending-semantic-review" if opaque_columns else "name-screen-complete",
            "name_detected_columns": sensitive_candidates,
            "opaque_columns_requiring_semantic_review": opaque_columns,
            "name_screen_is_conclusive": False,
        },
        "columns": column_profiles,
    }


def dictionary_rows(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sheet in profiles:
        for column in sheet["columns"]:
            rows.append(
                {
                    "sheet": column["sheet"],
                    "column": column["column"],
                    "pandas_dtype": column["pandas_dtype"],
                    "inferred_type": column["inferred_type"],
                    "non_null_count": column["non_null_count"],
                    "missing_count": column["missing_count"],
                    "missing_rate": column["missing_rate"],
                    "unique_count": column["unique_count"],
                    "constant": column["constant"],
                    "near_constant": column["near_constant"],
                    "mixed_python_types": column["mixed_python_types"],
                    "sensitive_name_candidate": column["sensitive_name_candidate"],
                    "sensitive_review_status": column["sensitive_review_status"],
                    "sample_values": " | ".join(str(value) for value in column["sample_values"]),
                }
            )
    return rows


def context_sheet_candidates(tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sheet_name, frame in tables.items():
        columns = [str(column) for column in frame.columns]
        matching_columns = [column for column in columns if CONTEXT_COLUMN_RE.search(column)]
        name_match = bool(CONTEXT_SHEET_RE.search(sheet_name))
        has_field_column = any(CONTEXT_FIELD_COLUMN_RE.search(column) for column in columns)
        has_description_column = any(
            CONTEXT_DESCRIPTION_COLUMN_RE.search(column) for column in columns
        )
        if not name_match and not (has_field_column and has_description_column):
            continue
        preview_frame = frame.iloc[:50, :12].copy()
        preview_frame.columns = [str(column) for column in preview_frame.columns]
        preview = [
            {column: scalar(value) for column, value in row.items()}
            for row in preview_frame.to_dict(orient="records")
        ]
        reasons = []
        if name_match:
            reasons.append("sheet-name")
        if matching_columns:
            reasons.append("metadata-columns")
        candidates.append(
            {
                "sheet": sheet_name,
                "reasons": reasons,
                "columns": columns,
                "matching_metadata_columns": matching_columns,
                "preview_row_count": len(preview),
                "preview": preview,
            }
        )
    return candidates


def markdown_report(
    source: Path,
    all_sheets: list[str],
    profiles: list[dict[str, Any]],
    table_read_spec: dict[str, Any],
) -> str:
    lines = [
        "# 表格数据体检",
        "",
        f"- 源文件：`{source.name}`",
        f"- 文件类型：`{source.suffix.lower()}`",
        f"- 工作表：{', '.join(all_sheets)}",
        "- 说明：本报告只读取原文件，没有修改数据。",
        "",
        "## 表格结构识别",
        "",
    ]
    for sheet_spec in table_read_spec["sheets"]:
        selected = sheet_spec["selected"]
        header_rows = selected["header_rows"]
        header_label = "无表头" if not header_rows else "、".join(f"第 {row} 行" for row in header_rows)
        lines.extend(
            [
                f"### 工作表：{sheet_spec['sheet']}",
                f"- 状态：`{sheet_spec['status']}`",
                f"- 当前候选表头：{header_label}",
                f"- 数据起始行：第 {selected['data_start_row']} 行",
                f"- 置信度：`{sheet_spec['confidence']}`",
                "- 候选列名：" + "、".join(f"`{name}`" for name in selected["column_names"][:12]),
            ]
        )
        if sheet_spec["requires_user_confirmation"]:
            lines.append("- 结论：检测到非标准或不确定表头；确认结构前不得进入研究问题规划。")
        else:
            lines.append("- 结论：标准单行表头已通过只读结构检查。")
        lines.append("")
    for profile in profiles:
        lines.extend(
            [
                f"## 工作表：{profile['sheet']}",
                "",
                f"- 规模：{profile['row_count']} 行 × {profile['column_count']} 列",
                f"- 重复行：{profile['duplicate_row_count']}",
                f"- 全空行：{profile['fully_empty_row_count']}",
                f"- 全空列：{profile['fully_empty_column_count']}",
                "",
                "| 字段 | 推断类型 | 缺失率 | 唯一值 | 常量 | 混合类型 | 敏感候选 |",
                "|---|---|---:|---:|---|---|---|",
            ]
        )
        for column in profile["columns"]:
            lines.append(
                "| {column} | {kind} | {missing:.1%} | {unique} | {constant} | "
                "{mixed} | {sensitive} |".format(
                    column=str(column["column"]).replace("|", "\\|"),
                    kind=column["inferred_type"],
                    missing=column["missing_rate"],
                    unique=column["unique_count"],
                    constant="是" if column["constant"] else "否",
                    mixed="是" if column["mixed_python_types"] else "否",
                    sensitive=(
                        "是（按名称）"
                        if column["sensitive_name_candidate"]
                        else "待核对字段含义"
                        if column["sensitive_review_status"] == "pending-semantic-review"
                        else "名称未命中"
                    ),
                )
            )
        issues = []
        for column in profile["columns"]:
            if column["missing_rate"] > 0:
                issues.append(f"`{column['column']}` 缺失率 {column['missing_rate']:.1%}")
            if column["constant"]:
                issues.append(f"`{column['column']}` 为常量或全空字段")
            elif column["near_constant"]:
                issues.append(f"`{column['column']}` 接近常量")
            if column["mixed_python_types"]:
                issues.append(f"`{column['column']}` 包含混合类型值")
            if column["sensitive_name_candidate"]:
                issues.append(f"`{column['column']}` 疑似敏感字段，样例已隐藏")
            elif column["sensitive_review_status"] == "pending-semantic-review":
                issues.append(
                    f"`{column['column']}` 的列名不含字段含义，敏感性状态待语义确认"
                )
            numeric = column.get("numeric_summary", {})
            if numeric.get("iqr_outlier_count", 0):
                issues.append(
                    f"`{column['column']}` 有 {numeric['iqr_outlier_count']} 个 IQR 规则候选异常值"
                )
        lines.extend(["", "### 需要确认的数据质量问题", ""])
        if issues:
            lines.extend(f"- {issue}" for issue in issues)
        else:
            lines.append("- 未发现基于当前规则的明显问题。")
        lines.extend(["", "候选异常值只是提示，不能在未经确认的情况下删除。", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.exists() or not source.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        print(
            f"Unsupported file type {source.suffix!r}; expected CSV, TSV, XLSX, or XLS.",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir.resolve()
    if source == output_dir or output_dir in source.parents:
        print("Output directory must not contain or replace the source file.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        explicit_header_rows = parse_header_rows(args.header_rows)
        raw_tables, all_sheet_names = load_raw_tables(source, args.sheet)
        tables, table_read_spec = build_structure_spec(
            source,
            raw_tables,
            explicit_header_rows,
            bool(args.headerless),
            bool(args.structure_confirmed),
        )
    except (ImportError, OSError, ValueError) as exc:
        print(f"Table profiling failed: {exc}", file=sys.stderr)
        return 2
    profiles = [profile_sheet(name, frame) for name, frame in tables.items()]
    payload = {
        "source_file": source.name,
        "source_suffix": source.suffix.lower(),
        "available_sheets": all_sheet_names,
        "profiled_sheets": list(tables),
        "generated_at": datetime.now().astimezone().isoformat(),
        "read_only": True,
        "table_structure_status": (
            "confirmation-required"
            if any(item["requires_user_confirmation"] for item in table_read_spec["sheets"])
            else "confirmed"
        ),
        "table_read_spec": table_read_spec,
        "profiles": profiles,
        "context_sheet_candidates": context_sheet_candidates(tables),
    }

    json_path = output_dir / "data-profile.json"
    csv_path = output_dir / "data-dictionary.csv"
    md_path = output_dir / "data-profile.md"
    spec_path = output_dir / "table-read-spec.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    spec_path.write_text(
        json.dumps(table_read_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(dictionary_rows(profiles)).to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(
        markdown_report(source, all_sheet_names, profiles, table_read_spec), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "source": source.name,
                "profiled_sheets": list(tables),
                "structure_confirmation_required": any(
                    item["requires_user_confirmation"] for item in table_read_spec["sheets"]
                ),
                "pending_structure_sheets": [
                    item["sheet"]
                    for item in table_read_spec["sheets"]
                    if item["requires_user_confirmation"]
                ],
                "outputs": [str(md_path), str(json_path), str(csv_path), str(spec_path)],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
