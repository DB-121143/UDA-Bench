"""
SQL 预处理使用说明
----------------
用途：将包含多条 SQL 的文件拆分成逐条编号的目录与 `sql.json`，便于后续放置 result.csv 并执行评测。

命令示例：
python3 -m evaluation.sql_preprocessor \
  --dataset Player \
  --task Select \
  --sql-file Query/Player/Select/select_queries_player.sql \
  --attributes-file Query/Player/Player_attributes.json \
  --output-root evaluation/demo_acc_result \
  --create-placeholder

python3 -m evaluation.sql_preprocessor \
  --dataset Player \
  --task Filter \
  --sql-file Query/Player/Filter/filter_queries_player.sql \
  --attributes-file Query/Player/Player_attributes.json \
  --output-root evaluation/demo_acc_result \
  --create-placeholder  

python3 -m evaluation.sql_preprocessor \
  --dataset Player \
  --task Agg \
  --sql-file Query/Player/Agg/agg_queries.sql \
  --attributes-file Query/Player/Player_attributes.json \
  --output-root evaluation/demo_acc_result 

主要参数：
- --dataset：数据集名称，对应 Query/{dataset}。
- --task：任务名（Select/Filter/Agg/Join/Mixed），用于输出路径拼接。
- --sql-file：包含多条 SQL 的文件。
- --attributes-file：对应的 *_attributes.json。
- --output-root：生成目录根路径，默认 evaluation/demo_acc_result。
- --create-placeholder：是否为每个 query 生成空的 result.csv 以便填充。

输出结构：
evaluation/demo_acc_result/{dataset}/{task}/{sql_stem}/{idx}/sql.json
（如加 --create-placeholder，会同时生成 result.csv）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import sqlglot

from .config import load_json
from .logging_utils import setup_logger
from .sql_parser import ParsedQuery, SqlParser
from .utils import ensure_dir


class SqlPreprocessor:
    """Split multi-SQL files, filter attributes, and lay out per-query folders."""

    def __init__(self, parser: Optional[SqlParser] = None) -> None:
        self.parser = parser or SqlParser()
        self.logger = setup_logger("sql_preprocessor")

    def split_sql_file(
        self,
        sql_path: Path,
        dataset: str,
        task: str,
        output_root: Path,
        attributes_path: Path,
        create_placeholder: bool = False,
    ) -> List[Path]:
        sql_text = Path(sql_path).read_text(encoding="utf-8")
        statements = sqlglot.parse(sql_text, error_level="ignore")
        attributes = load_json(attributes_path)
        query_dirs: List[Path] = []

        for idx, stmt in enumerate(statements, start=1):
            sql_str = stmt.sql()
            parsed = self.parser.parse(sql_str)
            payload = self._build_payload(sql_str, parsed, attributes)
            query_dir = Path(output_root) / dataset / task / Path(sql_path).stem / str(idx)
            ensure_dir(query_dir)
            out_file = query_dir / "sql.json"
            out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if create_placeholder:
                (query_dir / "result.csv").touch(exist_ok=True)
            query_dirs.append(query_dir)
            self.logger.info("Prepared query %s under %s", idx, query_dir)

        return query_dirs

    def _build_payload(self, sql: str, parsed: ParsedQuery, attributes: Mapping[str, Mapping]) -> Dict:
        payload: Dict = {"sql": sql}
        needed = self._collect_needed_columns(parsed)
        for table, cols in needed.items():
            if table not in attributes:
                continue
            table_meta = {}
            for col in cols:
                if col in attributes[table]:
                    table_meta[col] = {
                        "value_type": attributes[table][col].get("value_type", "str"),
                        "description": attributes[table][col].get("description"),
                    }
            if table_meta:
                payload[table] = table_meta
        return payload

    def _collect_needed_columns(self, parsed: ParsedQuery) -> Dict[str, List[str]]:
        needed: Dict[str, List[str]] = {}
        for item in parsed.select_items:
            if item.table and item.column:
                needed.setdefault(item.table, []).append(item.column)
            elif item.column:
                needed.setdefault(parsed.tables[0] if parsed.tables else "unknown", []).append(item.column)
        for gb in parsed.group_by:
            if "." in gb:
                table, col = gb.split(".", 1)
                needed.setdefault(table, []).append(col)
            elif parsed.tables:
                needed.setdefault(parsed.tables[0], []).append(gb)
        return {t: sorted(set(cols)) for t, cols in needed.items()}


def main():
    parser = argparse.ArgumentParser(description="Split SQL files and create per-query manifests.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., Player")
    parser.add_argument("--task", required=True, help="Task folder name, e.g., Select or Filter")
    parser.add_argument("--sql-file", required=True, type=Path, help="Path to the SQL file containing one or more queries")
    parser.add_argument("--attributes-file", required=True, type=Path, help="Path to *_attributes.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/demo_acc_result"),
        help="Root directory for generated query folders",
    )
    parser.add_argument("--create-placeholder", action="store_true", help="Create empty result.csv files for convenience")
    args = parser.parse_args()

    preprocessor = SqlPreprocessor()
    preprocessor.split_sql_file(
        sql_path=args.sql_file,
        dataset=args.dataset,
        task=args.task,
        output_root=args.output_root,
        attributes_path=args.attributes_file,
        create_placeholder=args.create_placeholder,
    )


if __name__ == "__main__":
    main()
