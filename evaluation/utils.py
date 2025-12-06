from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd

WHITESPACE_PATTERN = re.compile(r"\s+")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not str(c).startswith("Unnamed")]
    return df.loc[:, keep]


def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces for string values:
    - trim leading/trailing whitespace
    - collapse multiple whitespaces to one
    - treat '||' as a standalone token (keeps separators readable)
    """
    text = str(text)
    text = text.replace("||", " || ")
    text = text.strip()
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].apply(lambda x: normalize_whitespace(x) if isinstance(x, str) else x)
    return df


def standardize_column_name(name: str) -> str:
    """Remove quotes/backticks and strip whitespace."""
    cleaned = str(name).strip()
    if cleaned.startswith(("`", '"')) and cleaned.endswith(("`", '"')):
        cleaned = cleaned[1:-1]
    return cleaned


def add_missing_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df


def normalize_file_id(value: Any) -> str:
    """Clean file_name-like id to bare stem."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    basename = Path(text).name
    if "." in basename:
        basename = basename.rsplit(".", 1)[0]
    return basename


def normalize_file_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert file_name/filename columns to id columns by stripping suffix.
    Supports {table}.file_name style.
    """
    rename_map: MutableMapping[str, str] = {}
    for col in list(df.columns):
        lower = str(col).lower()
        if lower in {"file_name", "filename"}:
            rename_map[col] = "id"
        elif lower.endswith(".file_name") or lower.endswith(".filename"):
            prefix = col.split(".", 1)[0]
            rename_map[col] = f"{prefix}.id"
    if rename_map:
        df = df.rename(columns=rename_map)
        for new_col in rename_map.values():
            df[new_col] = df[new_col].apply(normalize_file_id)
    return df


def split_multi_value(cell: Any, sep: str = "||") -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    raw = normalize_whitespace(cell)
    if raw == "":
        return []
    return [part.strip() for part in raw.split(sep) if part.strip()]


def coerce_numeric(value: Any, value_type: str) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    if value_type == "int":
        try:
            return int(str(value).strip())
        except Exception:
            return value
    if value_type == "float":
        try:
            return float(str(value).strip())
        except Exception:
            return value
    return value


def normalize_types(df: pd.DataFrame, attributes: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> pd.DataFrame:
    """
    Cast columns according to value_type metadata when possible.
    attributes format: {table: {attr: {"value_type": ...}}}
    """
    df = df.copy()
    for table, cols in attributes.items():
        for col, meta in cols.items():
            value_type = meta.get("value_type")
            if value_type and col in df.columns:
                df[col] = df[col].apply(lambda x: coerce_numeric(x, value_type))
            dotted = f"{table}.{col}"
            if dotted in df.columns:
                df[dotted] = df[dotted].apply(lambda x: coerce_numeric(x, value_type))
    return df


def format_primary_key(row: pd.Series, keys: Sequence[str]) -> tuple:
    return tuple(str(row[k]) for k in keys)
