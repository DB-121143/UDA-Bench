from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from .logging_utils import setup_logger
from .utils import (
    add_missing_columns,
    clean_string_columns,
    drop_unnamed_columns,
    normalize_file_name_columns,
    normalize_types,
    standardize_column_name,
)


class ResultLoader:
    """Load and normalize system output CSV."""

    def __init__(
        self,
        expected_columns: Sequence[str],
        stop_columns: Sequence[str],
        attributes: Mapping,
        logger_name: str = "result_loader",
    ) -> None:
        self.expected_columns = list(expected_columns)
        self.stop_columns = list(stop_columns)
        self.attributes = attributes
        self.logger = setup_logger(logger_name)

    def load(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result CSV not found: {path}")
        df = pd.read_csv(path)
        df = drop_unnamed_columns(df)
        renamed = {col: standardize_column_name(col) for col in df.columns}
        df = df.rename(columns=renamed)
        df = normalize_file_name_columns(df)
        df = add_missing_columns(df, self.expected_columns)
        df = add_missing_columns(df, self.stop_columns)
        df = clean_string_columns(df)
        df = normalize_types(df, self.attributes)
        df = df.loc[:, self._column_order(df.columns)]
        self.logger.info("Loaded result from %s with %s rows", path, len(df))
        return df

    def _column_order(self, cols: Sequence[str]):
        ordered = []
        for col in self.expected_columns:
            if col in cols:
                ordered.append(col)
        for col in cols:
            if col not in ordered:
                ordered.append(col)
        return ordered
