from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import sqlglot
from sqlglot import exp


@dataclass
class SelectItem:
    raw: str
    output_name: str
    table: Optional[str]
    column: Optional[str]
    agg_func: Optional[str] = None
    is_agg: bool = False

    @property
    def normalized_output(self) -> str:
        return self.output_name

    @property
    def source_name(self) -> str:
        if self.table and self.column:
            return f"{self.table}.{self.column}"
        return self.column or self.output_name


@dataclass
class ParsedQuery:
    sql: str
    tables: List[str]
    select_items: List[SelectItem]
    group_by: List[str]
    join_keys: List[str]
    stop_columns: List[str]
    query_type: str
    primary_keys: List[str] = field(default_factory=list)

    @property
    def output_columns(self) -> List[str]:
        return [item.normalized_output for item in self.select_items]


class SqlParser:
    """Lightweight sqlglot-based parser to extract schema hints."""

    def parse(self, sql: str) -> ParsedQuery:
        expr = sqlglot.parse_one(sql, error_level="ignore")
        tables = self._collect_tables(expr)
        select_items = [self._parse_select_item(item) for item in expr.selects]
        group_by = self._collect_group_by(expr)
        join_keys = self._collect_join_keys(expr)
        stop_columns = self._detect_stop_columns(select_items, tables)
        query_type = self._detect_query_type(group_by, select_items, tables)
        primary_keys = self._infer_primary_keys(query_type, group_by, tables, join_keys)
        return ParsedQuery(
            sql=sql,
            tables=tables,
            select_items=select_items,
            group_by=group_by,
            join_keys=join_keys,
            stop_columns=stop_columns,
            query_type=query_type,
            primary_keys=primary_keys,
        )

    def _collect_tables(self, expr: exp.Expression) -> List[str]:
        tables: List[str] = []
        for table in expr.find_all(exp.Table):
            name = table.name
            if name not in tables:
                tables.append(name)
        return tables

    def _collect_group_by(self, expr: exp.Expression) -> List[str]:
        group_by_expr = expr.args.get("group")
        if not group_by_expr:
            return []
        cols: List[str] = []
        for node in group_by_expr.expressions:
            if isinstance(node, exp.Column):
                cols.append(self._column_identifier(node))
            else:
                cols.append(node.sql())
        return cols

    def _collect_join_keys(self, expr: exp.Expression) -> List[str]:
        keys: List[str] = []
        for join in expr.find_all(exp.Join):
            condition = join.args.get("on")
            if not condition:
                continue
            for eq_expr in condition.find_all(exp.EQ):
                left, right = eq_expr.left, eq_expr.right
                if isinstance(left, exp.Column):
                    keys.append(self._column_identifier(left))
                if isinstance(right, exp.Column):
                    keys.append(self._column_identifier(right))
        return sorted(set(keys))

    def _detect_stop_columns(self, items: Sequence[SelectItem], tables: Sequence[str]) -> List[str]:
        stop_cols: List[str] = []
        for item in items:
            name = item.output_name.lower()
            source = (item.source_name or "").lower()
            if name == "id" or source.endswith(".id"):
                stop_cols.append(item.output_name)
            if source.endswith(".id") and item.output_name not in stop_cols:
                stop_cols.append(item.output_name)
        for table in tables:
            candidate = f"{table}.id"
            if candidate not in stop_cols:
                stop_cols.append(candidate)
        if "id" not in stop_cols:
            stop_cols.append("id")
        return sorted(set(stop_cols))

    def _detect_query_type(self, group_by: Sequence[str], items: Sequence[SelectItem], tables: Sequence[str]) -> str:
        if group_by or any(i.is_agg for i in items):
            return "aggregation"
        if len(tables) > 1:
            return "join"
        return "select_filter"

    def _infer_primary_keys(
        self, query_type: str, group_by: Sequence[str], tables: Sequence[str], join_keys: Sequence[str]
    ) -> List[str]:
        if query_type == "aggregation":
            return list(group_by) if group_by else ["id"]
        if query_type == "join":
            if join_keys:
                return list(join_keys)
            return [f"{table}.id" for table in tables]
        return ["id"]

    def _parse_select_item(self, node: exp.Expression) -> SelectItem:
        if isinstance(node, exp.Alias):
            output_name = node.alias
            value = node.this
        else:
            output_name = (
                getattr(node, "alias_or_name", None)
                or getattr(node, "output_name", None)
                or node.sql()
            )
            value = node

        table = None
        column = None
        agg_func: Optional[str] = None
        is_agg = False

        if isinstance(value, exp.Column):
            table = value.table
            column = value.name
        elif isinstance(value, exp.Func):
            is_agg = bool(getattr(value, "is_aggregate", False) or isinstance(value, exp.AggFunc))
            agg_func = value.sql_name().upper() if hasattr(value, "sql_name") else value.key.upper()
            col_ref = value.args.get("this")
            if isinstance(col_ref, exp.Column):
                table = col_ref.table
                column = col_ref.name

        return SelectItem(
            raw=node.sql(),
            output_name=output_name,
            table=table,
            column=column,
            agg_func=agg_func,
            is_agg=is_agg,
        )

    def _column_identifier(self, column: exp.Column) -> str:
        if column.table:
            return f"{column.table}.{column.name}"
        return column.name
