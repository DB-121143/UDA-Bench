from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

from .config import load_json
from .sql_parser import ParsedQuery, SqlParser
from .utils import standardize_column_name


@dataclass
class ColumnMeta:
    value_type: str
    description: Optional[str] = None


class QueryManifest:
    """
    Bundle SQL text, parsed query info, and attribute metadata for downstream evaluators.
    """

    def __init__(self, sql: str, parsed: ParsedQuery, attributes: Mapping[str, Mapping[str, Mapping]]) -> None:
        self.sql = sql
        self.parsed = parsed
        self.attributes = attributes

    @classmethod
    def from_files(cls, sql_file: Path, attributes_file: Path, parser: Optional[SqlParser] = None) -> "QueryManifest":
        sql_text = cls._load_sql(sql_file)
        parser = parser or SqlParser()
        parsed = parser.parse(sql_text)
        attributes = load_json(Path(attributes_file))
        return cls(sql_text, parsed, attributes)

    @staticmethod
    def _load_sql(sql_file: Path) -> str:
        content = Path(sql_file).read_text(encoding="utf-8")
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "sql" in data:
                return data["sql"]
        except Exception:
            pass
        return content

    def get_column_meta(self, output_name: str) -> Optional[ColumnMeta]:
        for item in self.parsed.select_items:
            if standardize_column_name(item.output_name) == standardize_column_name(output_name):
                table = item.table
                column = item.column
                if table and column:
                    meta = self.attributes.get(table, {}).get(column)
                    if meta:
                        return ColumnMeta(value_type=meta.get("value_type", "str"), description=meta.get("description"))
                if column:
                    # single table select without explicit table prefix
                    meta = self._lookup_column(column)
                    if meta:
                        return ColumnMeta(value_type=meta.get("value_type", "str"), description=meta.get("description"))
        # fallback: try by name
        meta = self._lookup_column(output_name)
        if meta:
            return ColumnMeta(value_type=meta.get("value_type", "str"), description=meta.get("description"))
        return None

    def _lookup_column(self, column: str) -> Optional[Mapping]:
        for table_meta in self.attributes.values():
            if column in table_meta:
                return table_meta[column]
        return None

    @property
    def stop_columns(self):
        return self.parsed.stop_columns

    @property
    def primary_keys(self):
        return self.parsed.primary_keys
