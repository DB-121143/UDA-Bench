#!/usr/bin/env python3
"""
Run SQL queries against the Player CSV ground-truth tables using DuckDB.

The script loads the GT CSVs, normalizes a few column names to match the
provided attribute metadata, executes every SQL statement in the query
folders, and writes each result to a per-folder `result/` directory that is
further nested by SQL filename and query number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "UDA-Bench" / "Query" / "Player"
ATTR_PATH = DATASET_ROOT / "Player_attributes.json"
LOG_PATH = Path(__file__).with_suffix(".txt")

TABLE_CSVS = {
    "city": "city.csv",
    "manager": "manager.csv",
    "player": "player.csv",
    "team": "team.csv",
}

# Column renames that cannot be inferred from the attribute JSON alone.
MANUAL_RENAMES = {
    "team": {
        "championship": "championships",
    },
}

QUERY_FOLDERS = ["Join", "Mixed", "Agg", "Select", "Filter"]


def load_attribute_lookup(attr_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Build a lookup of expected column names (lower-cased) per table.
    Example: {"player": {"name": "name", "nba_championships": "nba_championships"}, ...}
    """
    meta = json.loads(attr_path.read_text())
    lookup: Dict[str, Dict[str, str]] = {}
    for table, cols in meta.items():
        table_lookup = {col["name"].lower(): col["name"] for col in cols}
        lookup[table] = table_lookup
    return lookup


def rename_columns_to_match_attributes(
    con: duckdb.DuckDBPyConnection, table: str, expected_lookup: Dict[str, str]
) -> None:
    """Rename columns to match the attribute metadata and manual corrections."""
    info = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    existing_cols = {row[1] for row in info}

    renames: Dict[str, str] = {}
    for col in existing_cols:
        lower = col.lower()
        if lower in expected_lookup:
            target = expected_lookup[lower]
            if col != target:
                renames[col] = target

    for src, dst in MANUAL_RENAMES.get(table, {}).items():
        if src in existing_cols:
            renames[src] = dst

    for src, dst in renames.items():
        con.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{src}" TO "{dst}";')


def load_tables(con: duckdb.DuckDBPyConnection, attr_lookup: Dict[str, Dict[str, str]]) -> None:
    """Create DuckDB tables for each CSV and normalize column names."""
    for table, csv_name in TABLE_CSVS.items():
        csv_path = DATASET_ROOT / csv_name
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv_auto('{csv_path.as_posix()}', HEADER=TRUE);
            """
        )
        rename_columns_to_match_attributes(con, table, attr_lookup.get(table, {}))


def extract_query_id(comment_text: str) -> Optional[str]:
    """Return the numeric id after 'Query' in the comment, if present."""
    match = re.search(r"Query\s*([0-9]+)", comment_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_category_id(comment_text: str) -> Optional[str]:
    """Return the numeric id after 'Category' in the comment, if present."""
    match = re.search(r"Category\s*([0-9]+)", comment_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def parse_sql_file(path: Path) -> List[Dict[str, Optional[str]]]:
    """
    Parse a SQL file with `--` comments preceding each query.

    Returns a list of dicts with keys:
    - sql: SQL string without the trailing semicolon
    - comment: concatenated comment text
    - category: category number (if seen most recently)
    - query_id: query number (if present after 'Query')
    """
    queries: List[Dict[str, Optional[str]]] = []
    pending_comments: List[str] = []
    sql_lines: List[str] = []
    current_category: Optional[str] = None

    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("--"):
                text = stripped.lstrip("-").strip()
                pending_comments.append(text)
                cat_id = extract_category_id(text)
                if cat_id:
                    current_category = cat_id
                continue

            if stripped == "":
                # Keep intentional blank lines inside a statement.
                if sql_lines:
                    sql_lines.append("")
                continue

            sql_lines.append(line)
            if ";" in line:
                statement = "\n".join(sql_lines).strip().rstrip(";").strip()
                comment_text = " ".join(pending_comments).strip()
                queries.append(
                    {
                        "sql": statement,
                        "comment": comment_text,
                        "category": current_category,
                        "query_id": extract_query_id(comment_text),
                    }
                )
                pending_comments = []
                sql_lines = []

    return queries


def build_output_filename(
    query_info: Dict[str, Optional[str]], ordinal: int, used_names: Iterable[str]
) -> str:
    """Create a deterministic filename for a query result."""
    cat_id = query_info.get("category")
    query_id = query_info.get("query_id")

    if cat_id and query_id:
        base = f"category_{cat_id}_query_{query_id}.csv"
    elif query_id:
        base = f"query_{query_id}.csv"
    elif cat_id:
        base = f"category_{cat_id}.csv"
    else:
        base = f"query_{ordinal}.csv"

    name = base
    suffix = 1
    used_set = set(used_names)
    while name in used_set:
        suffix += 1
        name = base.replace(".csv", f"_{suffix}.csv")
    return name


def copy_query_to_csv(con: duckdb.DuckDBPyConnection, sql: str, output_path: Path) -> None:
    """Execute a query and persist the result as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_body = sql.strip().rstrip(";")
    con.execute(
        f"""
        COPY (
            {query_body}
        ) TO '{output_path.as_posix()}' (FORMAT CSV, HEADER TRUE);
        """
    )


def parse_primary_table(sql: str) -> Optional[str]:
    """Return the first table name after FROM, if present."""
    match = re.search(r"(?i)\bfrom\s+([a-zA-Z_][\w]*)", sql)
    if match:
        return match.group(1)
    return None


def has_groupby_clause(sql: str) -> bool:
    """Return True when the SQL text contains a GROUP BY clause."""
    return bool(re.search(r"(?i)\bgroup\s+by\b", sql))


def extract_tables(sql: str) -> List[str]:
    """Collect ordered, unique table names appearing after FROM/JOIN."""
    table_pattern = re.compile(r"(?i)\b(?:from|join)\s+([a-zA-Z_][\w]*)")
    seen = set()
    tables: List[str] = []
    for match in table_pattern.finditer(sql):
        table = match.group(1)
        lowered = table.lower()
        if lowered not in seen:
            seen.add(lowered)
            tables.append(table)
    return tables


def prepend_columns_after_select(sql: str, columns: List[str]) -> str:
    """Insert extra columns immediately after SELECT / SELECT DISTINCT."""
    if not columns:
        return sql

    select_match = re.match(r"(?is)^(\s*select\s+(?:distinct\s+)?)", sql)
    if not select_match:
        return sql

    prefix = select_match.group(1)
    remainder = sql[select_match.end():].lstrip()
    injected = ", ".join(columns)
    if remainder:
        return f"{prefix}{injected}, {remainder}"
    return f"{prefix}{injected}"


def add_id_columns(sql: str) -> str:
    """
    Add id columns to the SELECT list when appropriate.

    - Skip queries that contain GROUP BY (aggregated results).
    - For single-table queries (Select/Filter), prepend a bare id column.
    - For joins, prepend per-table id columns named as `{table}.id`.
    """
    if has_groupby_clause(sql):
        return sql

    tables = extract_tables(sql)
    if len(tables) <= 1:
        return prepend_columns_after_select(sql, ["id"])

    id_columns = [f'{table}.id AS "{table}.id"' for table in tables]
    return prepend_columns_after_select(sql, id_columns)


def missing_column_from_error(message: str) -> Optional[str]:
    """Extract the missing column name from a DuckDB binder error."""
    match = re.search(r'Referenced column "([^"]+)" not found in FROM clause', message)
    if match:
        return match.group(1)
    return None


def is_groupby_error(message: str) -> bool:
    """Return True when the error is about a missing GROUP BY column."""
    return "must appear in the GROUP BY clause" in message


def ensure_null_column(con: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    """Add a nullable VARCHAR column when it does not already exist."""
    cols = {row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    if column in cols:
        return False
    con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" VARCHAR;')
    return True


def log_entry(
    entries: List[str],
    sql_file: Path,
    query_info: Dict[str, Optional[str]],
    message: str,
) -> None:
    label_parts = []
    if query_info.get("category"):
        label_parts.append(f"category={query_info['category']}")
    if query_info.get("query_id"):
        label_parts.append(f"query={query_info['query_id']}")
    label = ", ".join(label_parts) if label_parts else "unlabeled"
    entries.append(f"{sql_file.name} [{label}] {message}")


def run_queries_in_folder(
    con: duckdb.DuckDBPyConnection, folder: Path, log: List[str]
) -> None:
    """Execute every SQL statement in all .sql files under `folder`."""
    sql_files = sorted(folder.glob("*.sql"))
    if not sql_files:
        print(f"No SQL files found in {folder}")
        return

    for sql_file in sql_files:
        result_dir = folder / "result" / sql_file.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        used_names: List[str] = []
        queries = parse_sql_file(sql_file)
        for idx, query in enumerate(queries, start=1):
            filename = build_output_filename(query, idx, used_names)
            used_names.append(filename)
            out_path = result_dir / filename
            try:
                sql_to_run = add_id_columns(str(query["sql"]))
                copy_query_to_csv(con, sql_to_run, out_path)
                print(f"[OK] {sql_file.name} -> {out_path.name} | {query.get('comment')}")
            except Exception as exc:  # noqa: BLE001
                err_msg = str(exc)
                if is_groupby_error(err_msg):
                    log_entry(log, sql_file, query, f"GROUP BY error: {err_msg}")
                    print(f"[WARN] {sql_file.name} ({out_path.name}) GROUP BY error logged")
                    continue

                missing_col = missing_column_from_error(err_msg)
                if missing_col:
                    target_table = parse_primary_table(str(query["sql"])) or "player"
                    added = ensure_null_column(con, target_table, missing_col)
                    log_entry(
                        log,
                        sql_file,
                        query,
                        f"missing column '{missing_col}' on {target_table}; added_null_column={added}",
                    )
                    try:
                        copy_query_to_csv(con, str(query["sql"]), out_path)
                        print(
                            f"[OK] {sql_file.name} -> {out_path.name} | {query.get('comment')} (missing column filled with NULL)"
                        )
                    except Exception as rerun_exc:  # noqa: BLE001
                        log_entry(
                            log,
                            sql_file,
                            query,
                            f"failed after adding column '{missing_col}': {rerun_exc}",
                        )
                        print(f"[FAIL] {sql_file.name} ({out_path.name}): {rerun_exc}")
                    continue

                log_entry(log, sql_file, query, f"execution error: {err_msg}")
                print(f"[FAIL] {sql_file.name} ({out_path.name}): {err_msg}")


def main() -> None:
    attr_lookup = load_attribute_lookup(ATTR_PATH)
    con = duckdb.connect(database=":memory:")
    load_tables(con, attr_lookup)
    log_entries: List[str] = []

    for folder_name in QUERY_FOLDERS:
        folder_path = DATASET_ROOT / folder_name
        if not folder_path.exists():
            print(f"Skipping missing folder: {folder_path}")
            continue
        run_queries_in_folder(con, folder_path, log_entries)

    log_content = "\n".join(log_entries)
    LOG_PATH.write_text(log_content + ("\n" if log_entries else ""))
    if log_entries:
        print(f"Logged {len(log_entries)} issue(s) to {LOG_PATH}")
    else:
        print(f"No issues logged. Log file written to {LOG_PATH}")


if __name__ == "__main__":
    # 手动运行示例:
    #   /data/QUEST/jzshe/miniconda3/envs/quest/bin/python evaluator/run_sql_on_gt/gen_gt4sql.py
    main()
