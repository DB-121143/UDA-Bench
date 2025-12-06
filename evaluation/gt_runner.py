from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import duckdb
import pandas as pd

from .logging_utils import setup_logger
from .utils import clean_string_columns, drop_unnamed_columns, normalize_types, standardize_column_name


class GtRunner:
    """Register GT CSVs into duckdb and execute SQL."""

    def __init__(self, gt_dir: Path, attributes: Mapping, logger_name: str = "gt_runner") -> None:
        self.gt_dir = Path(gt_dir)
        self.attributes = attributes
        self.logger = setup_logger(logger_name)
        self.conn = duckdb.connect(database=":memory:")
        self._tables_loaded = False

    def _load_tables(self) -> None:
        for csv_path in sorted(self.gt_dir.glob("*.csv")):
            table = csv_path.stem
            df = pd.read_csv(csv_path)
            df = drop_unnamed_columns(df)
            df = clean_string_columns(df)
            df = normalize_types(df, self.attributes)
            self.conn.register(table, df)
            self.logger.debug("Registered table %s from %s", table, csv_path)
        self._tables_loaded = True

    def run(self, sql: str) -> pd.DataFrame:
        if not self._tables_loaded:
            self._load_tables()
        self.logger.info("Running SQL on GT")
        df = self.conn.execute(sql).fetchdf()
        df = self._post_process(df)
        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = {col: standardize_column_name(col) for col in df.columns}
        df = df.rename(columns=renamed)
        df = clean_string_columns(df)
        return df
