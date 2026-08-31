#!/usr/bin/env python3
"""Execute an approved data-preparation plan on a copy of a table."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    print(
        json.dumps(
            {
                "ok": False,
                "error": "pandas is required. Ask before installing dependencies.",
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from file_utils import sha256_file  # noqa: E402
from missingness_bias import build_missingness_screen, flatten_missingness_screen  # noqa: E402
from table_utils import unique_column_names  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an approved preparation plan without changing the source table."
    )
    parser.add_argument("--input", required=True, help="Source CSV, TSV, XLSX, or XLS file")
    parser.add_argument("--plan", required=True, help="Approved data-preparation plan JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned data and logs")
    parser.add_argument("--sheet", help="Excel sheet name when not specified by the plan")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's existing output files",
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
        fail(f"{label} JSON must contain an object")
    return value


def read_delimited(
    path: Path, sep: str | None, header: int | list[int] | None = 0
) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            if sep is None:
                return pd.read_csv(
                    path, sep=None, engine="python", encoding=encoding, header=header
                )
            return pd.read_csv(path, sep=sep, encoding=encoding, header=header)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    fail("Unable to decode delimited file: " + " | ".join(errors))
    raise AssertionError("unreachable")


def header_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_missing_tokens(frame: pd.DataFrame, tokens: list[str]) -> pd.DataFrame:
    """Apply the missing-value contract recorded by the profiler."""
    result = frame.copy()
    token_set = set(tokens)
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        mask = series.map(
            lambda value: isinstance(value, str) and value.strip() in token_set
        )
        if bool(mask.any()):
            result.loc[mask, column] = pd.NA
    return result


def apply_table_read_spec(raw: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    header_rows = spec.get("header_rows")
    data_start_row = spec.get("data_start_row")
    expected_names = spec.get("column_names")
    if not isinstance(header_rows, list) or not all(
        isinstance(row, int) and row >= 1 for row in header_rows
    ):
        fail("table_read_spec.header_rows must contain positive 1-based row numbers")
    if not isinstance(data_start_row, int) or data_start_row < 1:
        fail("table_read_spec.data_start_row must be a positive 1-based row number")
    if header_rows and data_start_row <= max(header_rows):
        fail("table_read_spec.data_start_row must be after all header rows")
    if not isinstance(expected_names, list) or len(expected_names) != raw.shape[1]:
        fail("table_read_spec.column_names does not match the source column count")
    missing_value_tokens = spec.get("missing_value_tokens", [])
    if not isinstance(missing_value_tokens, list) or not all(
        isinstance(token, str) and token for token in missing_value_tokens
    ):
        fail("table_read_spec.missing_value_tokens must contain non-empty strings")

    if header_rows:
        if max(header_rows) > len(raw):
            fail("Confirmed header row is outside the source table")
        names: list[str] = []
        for column_index in range(raw.shape[1]):
            parts = [header_text(raw.iat[row - 1, column_index]) for row in header_rows]
            names.append(" | ".join(part for part in parts if part))
        names = unique_column_names(names)
    else:
        names = [f"column_{index + 1}" for index in range(raw.shape[1])]
    if names != expected_names:
        fail("Source header no longer matches the confirmed table-read specification")

    frame = raw.iloc[data_start_row - 1 :].copy()
    frame.columns = names
    frame = normalize_missing_tokens(frame, missing_value_tokens)
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


def load_table(path: Path, plan: dict[str, Any], sheet_arg: str | None) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    spec = plan.get("table_read_spec")
    if not isinstance(spec, dict):
        fail("Approved plan has no confirmed table_read_spec; return to profiling")
    if spec.get("source_suffix") != suffix:
        fail("Source file type does not match the confirmed table-read specification")
    plan_sheet = spec.get("sheet")
    if not isinstance(plan_sheet, str) or not plan_sheet:
        fail("Confirmed table-read specification has no sheet name")
    if sheet_arg and sheet_arg != plan_sheet:
        fail("--sheet conflicts with the confirmed table-read specification")
    if suffix == ".csv":
        raw = read_delimited(path, None, header=None)
        return apply_table_read_spec(raw, spec), "CSV"
    if suffix == ".tsv":
        raw = read_delimited(path, "\t", header=None)
        return apply_table_read_spec(raw, spec), "TSV"
    if suffix in {".xlsx", ".xls"}:
        sheet: str | int = plan_sheet
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
            return apply_table_read_spec(raw, spec), str(sheet)
        except (ImportError, ValueError, OSError) as exc:
            fail(f"Unable to read Excel sheet {sheet!r}: {exc}")
    fail(f"Unsupported input format: {suffix}. Use CSV, TSV, XLSX, or XLS.")
    raise AssertionError("unreachable")


def source_rows(index: pd.Index) -> list[int]:
    rows: list[int] = []
    for value in index.tolist():
        try:
            # The dataframe keeps the raw, zero-based source-row index after
            # applying table_read_spec, so +1 yields the spreadsheet/CSV row.
            rows.append(int(value) + 1)
        except (TypeError, ValueError):
            fail("Internal row index is not traceable to the source table")
    return rows


def ensure_column(df: pd.DataFrame, column: Any, decision_id: str) -> str:
    if not isinstance(column, str) or not column:
        fail(f"Decision {decision_id!r} does not identify a column")
    if column not in df.columns:
        fail(f"Decision {decision_id!r} references missing column {column!r}")
    return column


def parameters_for(decision: dict[str, Any]) -> dict[str, Any]:
    value = decision.get("execution_parameters", {})
    if not isinstance(value, dict):
        fail(f"Decision {decision.get('id')!r} execution_parameters must be an object")
    return value


def base_action(decision: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    return {
        "decision_id": decision.get("id"),
        "topic": decision.get("topic"),
        "selected_option": decision.get("selected_option"),
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
        "affected_source_rows": [],
        "details": {},
    }


def finish_action(action: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    action["rows_after"] = int(len(df))
    action["columns_after"] = int(len(df.columns))
    action["affected_row_count"] = len(action["affected_source_rows"])
    return action


def execute_duplicate(
    df: pd.DataFrame, decision: dict[str, Any], action: dict[str, Any]
) -> pd.DataFrame:
    choice = decision["selected_option"]
    if choice == "keep":
        mask = df.duplicated(keep=False)
        action["details"] = {"duplicate_rows_detected": int(mask.sum()), "operation": "none"}
        return df
    if choice == "drop-exact-duplicates":
        mask = df.duplicated(keep="first")
        action["affected_source_rows"] = source_rows(df.index[mask])
        action["details"] = {"operation": "drop exact duplicates; keep first occurrence"}
        return df.loc[~mask].copy()
    if choice == "drop-fully-empty-rows":
        mask = df.isna().all(axis=1)
        action["affected_source_rows"] = source_rows(df.index[mask])
        action["details"] = {"operation": "drop fully empty rows"}
        return df.loc[~mask].copy()
    if choice == "deduplicate-with-key":
        keys = parameters_for(decision).get("key_columns")
        if not isinstance(keys, list) or not keys or not all(isinstance(x, str) for x in keys):
            fail("deduplicate-with-key requires approved execution_parameters.key_columns")
        missing = [column for column in keys if column not in df.columns]
        if missing:
            fail("Deduplication key columns are missing: " + ", ".join(missing))
        mask = df.duplicated(subset=keys, keep="first")
        action["affected_source_rows"] = source_rows(df.index[mask])
        action["details"] = {"operation": "deduplicate with key", "key_columns": keys}
        return df.loc[~mask].copy()
    fail(f"Unsupported duplicate-row choice: {choice}")
    raise AssertionError("unreachable")


def execute_missing(
    df: pd.DataFrame,
    decision: dict[str, Any],
    action: dict[str, Any],
    deferred: list[dict[str, Any]],
) -> pd.DataFrame:
    decision_id = str(decision["id"])
    column = ensure_column(df, decision.get("column") or decision_id.split(":", 1)[-1], decision_id)
    choice = decision["selected_option"]
    missing_mask = df[column].isna()
    action["details"]["column"] = column
    action["details"]["missing_before"] = int(missing_mask.sum())
    if choice == "complete-case":
        action["affected_source_rows"] = source_rows(df.index[missing_mask])
        action["details"]["operation"] = "drop rows missing this column"
        return df.loc[~missing_mask].copy()
    if choice == "median-imputation":
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid_nonmissing = df[column].notna() & numeric.isna()
        if invalid_nonmissing.any():
            fail(f"median-imputation requires a numeric column: {column}")
        median = numeric.median(skipna=True)
        if pd.isna(median):
            fail(f"Cannot compute a median because column {column!r} has no numeric values")
        action["affected_source_rows"] = source_rows(df.index[missing_mask])
        df = df.copy()
        df[column] = numeric.fillna(median)
        action["details"].update({"operation": "median imputation", "fill_value": float(median)})
        return df
    if choice == "most-frequent":
        observed = df.loc[~missing_mask, column]
        if observed.empty:
            fail(f"Cannot impute {column!r} because it has no observed values")
        modes = observed.mode(dropna=True)
        if modes.empty:
            fail(f"Cannot determine a most-frequent value for {column!r}")
        fill_value = modes.iloc[0]
        action["affected_source_rows"] = source_rows(df.index[missing_mask])
        df = df.copy()
        df[column] = df[column].fillna(fill_value)
        action["details"].update(
            {"operation": "most-frequent imputation", "fill_value": str(fill_value)}
        )
        return df
    if choice == "explicit-missing-category":
        params = parameters_for(decision)
        label = params.get("label", "[MISSING]")
        if not isinstance(label, str) or not label:
            fail("explicit-missing-category label must be non-empty text")
        action["affected_source_rows"] = source_rows(df.index[missing_mask])
        df = df.copy()
        df[column] = df[column].astype("object").where(~missing_mask, label)
        action["details"].update(
            {"operation": "explicit missing category", "label": label}
        )
        return df
    if choice == "missing-indicator":
        indicator = f"{column}__missing"
        if indicator in df.columns:
            fail(f"Missing-indicator output column already exists: {indicator}")
        action["affected_source_rows"] = source_rows(df.index[missing_mask])
        df = df.copy()
        df[indicator] = missing_mask.astype("int8")
        action["details"].update(
            {"operation": "add missingness indicator; retain original missing values", "indicator": indicator}
        )
        deferred.append(
            {"decision_id": decision_id, "reason": f"Original missing values in {column} remain for model-specific handling"}
        )
        return df
    if choice == "model-specific-method":
        action["details"]["operation"] = "deferred to model-specific workflow"
        deferred.append(
            {"decision_id": decision_id, "reason": f"Missing values in {column} were explicitly deferred"}
        )
        return df
    if choice == "multiple-imputation":
        fail(
            "multiple-imputation cannot be represented by one cleaned-data.csv; "
            "return to confirmation or add a separate multiple-imputation workflow"
        )
    fail(f"Unsupported missing-value choice: {choice}")
    raise AssertionError("unreachable")


def numeric_series(df: pd.DataFrame, column: str, decision_id: str) -> pd.Series:
    numeric = pd.to_numeric(df[column], errors="coerce")
    invalid_nonmissing = df[column].notna() & numeric.isna()
    if invalid_nonmissing.any():
        fail(f"Decision {decision_id!r} requires numeric column {column!r}")
    return numeric


def execute_outlier(
    df: pd.DataFrame,
    decision: dict[str, Any],
    action: dict[str, Any],
    deferred: list[dict[str, Any]],
) -> pd.DataFrame:
    decision_id = str(decision["id"])
    column = ensure_column(df, decision.get("column") or decision_id.split(":", 1)[-1], decision_id)
    choice = decision["selected_option"]
    action["details"]["column"] = column
    if choice == "keep-and-run-diagnostics":
        action["details"]["operation"] = "keep values; defer influence diagnostics"
        deferred.append(
            {"decision_id": decision_id, "reason": f"Run model diagnostics for candidate outliers in {column}"}
        )
        return df

    numeric = numeric_series(df, column, decision_id)
    params = parameters_for(decision)
    if choice == "transform":
        method = params.get("method")
        output_column = params.get("output_column", f"{column}__{method}")
        if method not in {"log1p", "sqrt"}:
            fail("transform requires approved execution_parameters.method of log1p or sqrt")
        if not isinstance(output_column, str) or not output_column or output_column in df.columns:
            fail("transform requires a new, non-empty execution_parameters.output_column")
        if method == "log1p" and (numeric.dropna() <= -1).any():
            fail(f"log1p requires all non-missing {column!r} values to be greater than -1")
        if method == "sqrt" and (numeric.dropna() < 0).any():
            fail(f"sqrt requires all non-missing {column!r} values to be non-negative")
        df = df.copy()
        df[output_column] = numeric.map(
            lambda value: math.log1p(value) if method == "log1p" and pd.notna(value)
            else (math.sqrt(value) if method == "sqrt" and pd.notna(value) else value)
        )
        action["details"].update(
            {"operation": "add transformed copy", "method": method, "output_column": output_column}
        )
        return df
    if choice == "winsorize":
        lower = params.get("lower_quantile")
        upper = params.get("upper_quantile")
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            fail("winsorize requires approved lower_quantile and upper_quantile")
        if not 0 <= float(lower) < float(upper) <= 1:
            fail("winsorize quantiles must satisfy 0 <= lower < upper <= 1")
        lower_value = float(numeric.quantile(float(lower)))
        upper_value = float(numeric.quantile(float(upper)))
        clipped = numeric.clip(lower=lower_value, upper=upper_value)
        changed = numeric.notna() & clipped.ne(numeric)
        action["affected_source_rows"] = source_rows(df.index[changed])
        df = df.copy()
        df[column] = clipped
        action["details"].update(
            {
                "operation": "winsorize",
                "lower_quantile": float(lower),
                "upper_quantile": float(upper),
                "lower_value": lower_value,
                "upper_value": upper_value,
            }
        )
        return df
    if choice == "exclude-with-domain-rule":
        lower = params.get("min")
        upper = params.get("max")
        if lower is None and upper is None:
            fail("exclude-with-domain-rule requires approved min and/or max")
        keep = numeric.notna()
        if lower is not None:
            if not isinstance(lower, (int, float)):
                fail("Domain-rule min must be numeric")
            keep &= numeric >= float(lower)
        if upper is not None:
            if not isinstance(upper, (int, float)):
                fail("Domain-rule max must be numeric")
            keep &= numeric <= float(upper)
        keep |= numeric.isna()
        action["affected_source_rows"] = source_rows(df.index[~keep])
        action["details"].update({"operation": "exclude outside domain", "min": lower, "max": upper})
        return df.loc[keep].copy()
    fail(f"Unsupported candidate-outlier choice: {choice}")
    raise AssertionError("unreachable")


def execute_encoding(
    df: pd.DataFrame,
    decision: dict[str, Any],
    action: dict[str, Any],
    metadata: dict[str, list[str]],
) -> pd.DataFrame:
    decision_id = str(decision["id"])
    column = ensure_column(df, decision.get("column") or decision_id.split(":", 1)[-1], decision_id)
    choice = decision["selected_option"]
    action["details"]["column"] = column
    if choice == "treat-as-categorical":
        metadata["categorical_columns"].append(column)
        action["details"].update(
            {"operation": "record as categorical for modeling", "category_count": int(df[column].nunique(dropna=True))}
        )
        return df
    if choice == "keep-as-text-and-exclude":
        metadata["analysis_exclusions"].append(column)
        action["details"]["operation"] = "preserve column but exclude from modeling"
        return df
    if choice == "manual-recode":
        params = parameters_for(decision)
        mapping = params.get("mapping")
        unmapped = params.get("unmapped", "error")
        if not isinstance(mapping, dict) or not mapping:
            fail("manual-recode requires approved execution_parameters.mapping")
        if unmapped not in {"error", "keep"}:
            fail("manual-recode execution_parameters.unmapped must be error or keep")
        source_as_text = df[column].map(lambda value: str(value) if pd.notna(value) else value)
        known = source_as_text.isna() | source_as_text.isin(mapping.keys())
        if unmapped == "error" and (~known).any():
            fail(f"manual-recode for {column!r} has unmapped non-missing values")
        changed = source_as_text.notna() & source_as_text.isin(mapping.keys())
        action["affected_source_rows"] = source_rows(df.index[changed])
        df = df.copy()
        df[column] = source_as_text.map(
            lambda value: mapping.get(value, value) if pd.notna(value) else value
        )
        metadata["categorical_columns"].append(column)
        action["details"].update(
            {"operation": "manual recode", "mapping_entry_count": len(mapping), "unmapped": unmapped}
        )
        return df
    fail(f"Unsupported categorical choice: {choice}")
    raise AssertionError("unreachable")


def convert_series(series: pd.Series, target_type: str) -> pd.Series:
    if target_type == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if target_type == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if target_type in {"text", "categorical"}:
        return series.map(lambda value: str(value) if pd.notna(value) else value)
    fail("type conversion target_type must be numeric, datetime, text, or categorical")
    raise AssertionError("unreachable")


def execute_type_conversion(
    df: pd.DataFrame,
    decision: dict[str, Any],
    action: dict[str, Any],
    metadata: dict[str, list[str]],
) -> pd.DataFrame:
    decision_id = str(decision["id"])
    column = ensure_column(df, decision.get("column") or decision_id.split(":", 1)[-1], decision_id)
    choice = decision["selected_option"]
    action["details"]["column"] = column
    if choice == "exclude":
        metadata["analysis_exclusions"].append(column)
        action["details"]["operation"] = "preserve column but exclude from modeling"
        return df

    params = parameters_for(decision)
    target_type = params.get("target_type")
    if not isinstance(target_type, str):
        fail(f"{choice} for {column!r} requires execution_parameters.target_type")
    source = df[column]
    if choice == "inspect-and-recode":
        mapping = params.get("mapping", {})
        if mapping:
            if not isinstance(mapping, dict):
                fail("inspect-and-recode execution_parameters.mapping must be an object")
            source = source.map(
                lambda value: mapping.get(str(value), value) if pd.notna(value) else value
            )
        converted = convert_series(source, target_type)
        invalid = source.notna() & converted.isna()
        if invalid.any():
            fail(
                f"inspect-and-recode for {column!r} still has {int(invalid.sum())} invalid values; "
                "revise the approved mapping or target_type"
            )
    elif choice == "coerce-invalid-to-missing":
        converted = convert_series(source, target_type)
        invalid = source.notna() & converted.isna()
    else:
        fail(f"Unsupported type-conversion choice: {choice}")

    df = df.copy()
    df[column] = converted
    action["affected_source_rows"] = source_rows(df.index[invalid])
    action["details"].update(
        {
            "operation": choice,
            "target_type": target_type,
            "invalid_value_count": int(invalid.sum()),
        }
    )
    if target_type == "categorical":
        metadata["categorical_columns"].append(column)
    return df


def execute_inclusion(
    df: pd.DataFrame,
    decision: dict[str, Any],
    action: dict[str, Any],
    metadata: dict[str, list[str]],
) -> pd.DataFrame:
    decision_id = str(decision["id"])
    column = ensure_column(df, decision.get("column") or decision_id.split(":", 1)[-1], decision_id)
    choice = decision["selected_option"]
    action["details"]["column"] = column
    if choice == "exclude":
        metadata["analysis_exclusions"].append(column)
        action["details"]["operation"] = "exclude from modeling"
        return df
    if choice == "include-with-justification":
        note = decision.get("user_note")
        if not isinstance(note, str) or not note.strip():
            fail("include-with-justification requires a non-empty approved user note")
        action["details"].update(
            {"operation": "retain for modeling", "justification": note.strip()}
        )
        return df
    fail(f"Unsupported inclusion choice: {choice}")
    raise AssertionError("unreachable")


def default_analysis_metadata(plan: dict[str, Any]) -> dict[str, list[str]]:
    metadata: dict[str, list[str]] = {"categorical_columns": [], "analysis_exclusions": []}
    for variable in plan.get("variable_actions", []):
        if not isinstance(variable, dict):
            continue
        column = variable.get("column")
        if not isinstance(column, str):
            continue
        encoding = variable.get("encoding")
        if isinstance(encoding, dict) and encoding.get("recommendation") == "treat-as-categorical":
            metadata["categorical_columns"].append(column)
        if variable.get("inclusion") == "exclude":
            metadata["analysis_exclusions"].append(column)
    return metadata


def missingness_variables(plan: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in plan.get("variable_actions", []):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        column = item.get("column")
        if role in {"outcome", "predictor", "control"} and isinstance(column, str):
            result.append(
                {
                    "role": role,
                    "column": column,
                    "inferred_type": str(item.get("inferred_type", "unknown")),
                }
            )
    if not result or not any(item["role"] == "outcome" for item in result):
        fail("Approved plan has no valid selected-variable roles for missingness review")
    return result


def validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "status": "approved",
        "requires_user_confirmation": False,
        "raw_data_unchanged": True,
        "data_preparation_executed": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            fail(f"Approved plan must contain {key}: {value!r}")
    decisions = plan.get("approved_decisions")
    if not isinstance(decisions, list):
        fail("Approved plan must contain an approved_decisions array")
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            fail("Every approved decision must be an object")
        decision_id = decision.get("id")
        if not isinstance(decision_id, str) or not decision_id:
            fail("Every approved decision must have an ID")
        if decision_id in seen:
            fail(f"Duplicate approved decision ID: {decision_id}")
        seen.add(decision_id)
        if decision.get("status") != "user-confirmed":
            fail(f"Decision {decision_id!r} is not user-confirmed")
        if not isinstance(decision.get("selected_option"), str):
            fail(f"Decision {decision_id!r} has no selected_option")
        if decision["selected_option"] not in decision.get("options", []):
            fail(f"Decision {decision_id!r} selected an option outside its approved options")
    return decisions


def check_source_plan_hash(plan: dict[str, Any]) -> dict[str, Any]:
    approval = plan.get("approval")
    if not isinstance(approval, dict):
        fail("Approved plan is missing approval provenance")
    expected = approval.get("source_plan_sha256")
    source = approval.get("source_plan")
    if not isinstance(expected, str) or len(expected) != 64:
        fail("Approved plan is missing a valid source-plan SHA-256")
    if not isinstance(source, str) or not source:
        fail("Approved plan is missing its source-plan path")
    source_path = Path(source)
    if not source_path.is_file():
        return {"status": "source-path-unavailable", "expected_sha256": expected}
    actual = sha256_file(source_path)
    if actual != expected:
        fail("Source draft plan hash no longer matches the approved plan")
    return {"status": "verified", "expected_sha256": expected, "actual_sha256": actual}


def write_outputs(
    df: pd.DataFrame,
    log: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path]:
    data_path = output_dir / "cleaned-data.csv"
    json_path = output_dir / "data-preparation-log.json"
    md_path = output_dir / "data-preparation-log.md"
    missingness_path = output_dir / "missingness-impact.csv"
    targets = [data_path, json_path, md_path, missingness_path]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        fail("Outputs already exist; use --overwrite only after approval: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=output_dir, suffix=".csv") as handle:
            data_temp = Path(handle.name)
        temp_paths.append(data_temp)
        df.reset_index(drop=True).to_csv(data_temp, index=False, encoding="utf-8-sig")
        os.replace(data_temp, data_path)
        temp_paths.remove(data_temp)

        log["output_sha256"] = sha256_file(data_path)
        pd.DataFrame(log.get("missingness_bias_rows", [])).to_csv(
            missingness_path, index=False, encoding="utf-8-sig"
        )
        json_text = json.dumps(log, ensure_ascii=False, indent=2) + "\n"
        md_text = build_markdown(log)
        for target, content in ((json_path, json_text), (md_path, md_text)):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", delete=False, dir=output_dir
            ) as handle:
                handle.write(content)
                temp = Path(handle.name)
            temp_paths.append(temp)
            os.replace(temp, target)
            temp_paths.remove(temp)
    finally:
        for temp in temp_paths:
            try:
                temp.unlink()
            except OSError:
                pass
    return data_path, json_path, md_path, missingness_path


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_markdown(log: dict[str, Any]) -> str:
    screen = log["missingness_bias_screen"]
    post = log["post_preparation_missingness"]
    lines = [
        "# 数据处理执行日志",
        "",
        f"- 状态：`{log['status']}`",
        f"- 执行时间：`{log['executed_at']}`",
        f"- 原始数据：`{log['source_file']}`",
        f"- 原始数据未改动：`{str(log['raw_data_unchanged']).lower()}`",
        f"- 行数：{log['input_rows']} → {log['output_rows']}",
        f"- 列数：{log['input_columns']} → {log['output_columns']}",
        f"- 完整案例情景：保留 {screen['complete_case_rows']} 行，排除 {screen['complete_case_excluded_rows']} 行（{screen['complete_case_excluded_rate']:.1%}）",
        f"- 处理后所选字段仍有缺失的记录：{post['rows_with_any_selected_missingness']} 行",
        "",
        "## 缺失值与样本构成",
        "",
        "处理前已比较缺失组与非缺失组的结果构成，完整数值见 `missingness-impact.csv`。这些比较不能证明缺失是随机的，也没有完成不同缺失处理方法下的模型系数敏感性分析。",
        f"当前结论范围：`{log['missingness_conclusion_contract']['scope']}`。",
        "",
        "## 已执行决定",
        "",
        "| 决定 ID | 用户选择 | 受影响行数 | 执行后行数 | 执行后列数 |",
        "|---|---|---:|---:|---:|",
    ]
    for action in log["actions"]:
        lines.append(
            "| {id} | {choice} | {affected} | {rows} | {columns} |".format(
                id=md_escape(action["decision_id"]),
                choice=md_escape(action["selected_option"]),
                affected=action["affected_row_count"],
                rows=action["rows_after"],
                columns=action["columns_after"],
            )
        )
    lines.extend(["", "## 延后到建模阶段的事项", ""])
    if log["deferred_actions"]:
        for item in log["deferred_actions"]:
            lines.append(f"- `{md_escape(item['decision_id'])}`：{md_escape(item['reason'])}")
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 边界说明",
            "",
            "本阶段只处理数据副本并生成日志；尚未拟合统计模型，也未生成最终报告。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    plan_path = Path(args.plan).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.is_file():
        fail(f"Source table does not exist: {input_path}")

    plan = read_json(plan_path, "approved plan")
    decisions = validate_plan(plan)
    source_name = plan.get("source_file")
    if isinstance(source_name, str) and Path(source_name).name.lower() != input_path.name.lower():
        fail(
            f"Input filename {input_path.name!r} does not match approved source {Path(source_name).name!r}"
        )
    source_plan_verification = check_source_plan_hash(plan)
    source_hash_before = sha256_file(input_path)
    table_read_spec = plan.get("table_read_spec")
    if not isinstance(table_read_spec, dict) or table_read_spec.get("source_sha256") != source_hash_before:
        fail("Source file hash does not match the confirmed table-read specification")
    df, sheet = load_table(input_path, plan, args.sheet)
    input_rows, input_columns = len(df), len(df.columns)
    variables = missingness_variables(plan)
    missingness_screen = build_missingness_screen(df, variables)
    approved_screen = plan.get("missingness_bias_screen")
    if not isinstance(approved_screen, dict) or approved_screen != missingness_screen:
        fail(
            "Missingness sample-composition evidence differs from the approved plan; "
            "return to planning instead of executing stale choices"
        )

    actions: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    metadata = default_analysis_metadata(plan)
    for decision in decisions:
        decision_id = str(decision["id"])
        action = base_action(decision, df)
        if decision_id.startswith("rows:"):
            df = execute_duplicate(df, decision, action)
        elif decision_id.startswith("missing:"):
            df = execute_missing(df, decision, action, deferred)
        elif decision_id.startswith("outlier:"):
            df = execute_outlier(df, decision, action, deferred)
        elif decision_id.startswith("encoding:"):
            df = execute_encoding(df, decision, action, metadata)
        elif decision_id.startswith("type:"):
            df = execute_type_conversion(df, decision, action, metadata)
        elif decision_id.startswith("inclusion:"):
            df = execute_inclusion(df, decision, action, metadata)
        else:
            fail(f"Unsupported approved decision type: {decision_id}")
        actions.append(finish_action(action, df))

    source_hash_after = sha256_file(input_path)
    if source_hash_after != source_hash_before:
        fail("Source table changed during execution; outputs were not written")
    status = "completed-with-deferred-actions" if deferred else "completed"
    selected_columns = [item["column"] for item in variables]
    absent_selected = [column for column in selected_columns if column not in df.columns]
    if absent_selected:
        fail("Selected columns disappeared during preparation: " + ", ".join(absent_selected))
    selected_missing = df[selected_columns].isna()
    post_preparation_missingness = {
        "rows": int(len(df)),
        "rows_with_any_selected_missingness": int(selected_missing.any(axis=1).sum()),
        "by_selected_column": {
            column: int(selected_missing[column].sum()) for column in selected_columns
        },
    }
    conclusion_contract = plan.get("missingness_conclusion_contract")
    if not isinstance(conclusion_contract, dict):
        fail("Approved plan is missing the missingness conclusion contract")
    log: dict[str, Any] = {
        "status": status,
        "data_preparation_executed": True,
        "modeling_executed": False,
        "raw_data_unchanged": True,
        "executed_at": datetime.now().astimezone().isoformat(),
        "source_file": str(input_path),
        "source_sheet": sheet,
        "table_read_spec": table_read_spec,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "approved_plan": str(plan_path),
        "approved_plan_sha256": sha256_file(plan_path),
        "source_plan_verification": source_plan_verification,
        "input_rows": int(input_rows),
        "input_columns": int(input_columns),
        "output_rows": int(len(df)),
        "output_columns": int(len(df.columns)),
        "actions": actions,
        "deferred_actions": deferred,
        "missingness_bias_screen": missingness_screen,
        "missingness_bias_rows": flatten_missingness_screen(missingness_screen),
        "post_preparation_missingness": post_preparation_missingness,
        "missingness_conclusion_contract": conclusion_contract,
        "analysis_metadata": {
            key: list(dict.fromkeys(values)) for key, values in metadata.items()
        },
    }
    data_path, json_path, md_path, missingness_path = write_outputs(
        df, log, output_dir, args.overwrite
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "input_rows": input_rows,
                "output_rows": len(df),
                "input_columns": input_columns,
                "output_columns": len(df.columns),
                "raw_data_unchanged": True,
                "modeling_executed": False,
                "outputs": [str(data_path), str(md_path), str(json_path), str(missingness_path)],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
