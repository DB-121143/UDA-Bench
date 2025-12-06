from __future__ import annotations

import pandas as pd

from .utils import normalize_file_name_columns


def normalize_result_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert file_name/filename columns to id columns by stripping suffixes.
    This is intended for system outputs where IDs are derived from file names.
    """
    return normalize_file_name_columns(df)
