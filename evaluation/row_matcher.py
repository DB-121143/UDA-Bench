from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .logging_utils import setup_logger
from .utils import format_primary_key


@dataclass
class MatchResult:
    gold_aligned: pd.DataFrame
    pred_aligned: pd.DataFrame
    len_gold: int
    len_pred: int
    matched_rows: int
    warnings: List[str]
    keys: List[str]


class RowMatcher:
    """Align rows based on primary keys, with optional secondary key for multi-entity cases."""

    def __init__(self, logger_name: str = "row_matcher") -> None:
        self.logger = setup_logger(logger_name)

    def match(
        self,
        gold_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        primary_keys: Sequence[str],
        secondary_key: Optional[str] = None,
    ) -> MatchResult:
        keys = list(primary_keys)
        if secondary_key and secondary_key not in keys:
            keys.append(secondary_key)

        self._ensure_columns_exist(gold_df, pred_df, keys)

        gold_norm = self._with_key_column(gold_df, keys)
        pred_norm = self._with_key_column(pred_df, keys)

        common_keys = sorted(set(gold_norm["__key"]) & set(pred_norm["__key"]))
        matched_gold_parts: List[pd.DataFrame] = []
        matched_pred_parts: List[pd.DataFrame] = []

        for key in common_keys:
            g_rows = gold_norm[gold_norm["__key"] == key]
            p_rows = pred_norm[pred_norm["__key"] == key]
            count = min(len(g_rows), len(p_rows))
            matched_gold_parts.append(g_rows.head(count).drop(columns="__key"))
            matched_pred_parts.append(p_rows.head(count).drop(columns="__key"))

        matched_gold = pd.concat(matched_gold_parts, ignore_index=True) if matched_gold_parts else gold_df.head(0)
        matched_pred = pd.concat(matched_pred_parts, ignore_index=True) if matched_pred_parts else pred_df.head(0)

        warnings = self._collect_warnings(gold_norm, pred_norm, keys)

        self.logger.info(
            "Aligned %s rows (gold=%s, pred=%s) using keys=%s",
            len(matched_gold),
            len(gold_df),
            len(pred_df),
            keys,
        )

        return MatchResult(
            gold_aligned=matched_gold,
            pred_aligned=matched_pred,
            len_gold=len(gold_df),
            len_pred=len(pred_df),
            matched_rows=len(matched_gold),
            warnings=warnings,
            keys=keys,
        )

    def _with_key_column(self, df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
        df = df.copy()
        for key in keys:
            df[key] = df[key].fillna("").astype(str)
        df["__key"] = df.apply(lambda r: format_primary_key(r, keys), axis=1)
        return df

    def _ensure_columns_exist(self, gold_df: pd.DataFrame, pred_df: pd.DataFrame, keys: Sequence[str]) -> None:
        missing_gold = [k for k in keys if k not in gold_df.columns]
        missing_pred = [k for k in keys if k not in pred_df.columns]
        if missing_gold:
            raise KeyError(f"Gold result missing key columns: {missing_gold}")
        if missing_pred:
            raise KeyError(f"Prediction missing key columns: {missing_pred}")

    def _collect_warnings(self, gold_df: pd.DataFrame, pred_df: pd.DataFrame, keys: Sequence[str]) -> List[str]:
        warnings: List[str] = []
        dup_gold = self._find_duplicates(gold_df, keys)
        dup_pred = self._find_duplicates(pred_df, keys)
        if dup_gold:
            warnings.append(f"Gold has duplicate keys for {dup_gold}")
        if dup_pred:
            warnings.append(f"Prediction has duplicate keys for {dup_pred}")
        return warnings

    def _find_duplicates(self, df: pd.DataFrame, keys: Sequence[str]) -> List[Tuple]:
        if not len(df):
            return []
        grp = df.groupby(list(keys)).size().reset_index(name="count")
        duplicates = grp[grp["count"] > 1]
        if duplicates.empty:
            return []
        return [tuple(row[key] for key in keys) for _, row in duplicates.iterrows()]
