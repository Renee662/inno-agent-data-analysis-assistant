"""Shared tabular-data helpers used by profiling and preparation."""

from __future__ import annotations


def unique_column_names(values: list[str]) -> list[str]:
    """Normalize blank labels and suffix duplicates with stable pandas-style names."""
    result: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values, 1):
        base = value.strip() or f"Unnamed: {index - 1}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        result.append(base if count == 0 else f"{base}.{count}")
    return result
