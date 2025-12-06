from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .comparators import LlmClient
from .config import EvalSettings
from .logging_utils import setup_logger
from .utils import format_primary_key, normalize_whitespace


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

    def __init__(
        self,
        logger_name: str = "row_matcher",
        settings: Optional[EvalSettings] = None,
        llm_client: Optional[LlmClient] = None,
    ) -> None:
        self.logger = setup_logger(logger_name)
        self.llm_client = llm_client or (LlmClient(settings, logger_name=f"{logger_name}.llm") if settings else None)

    def match(
        self,
        gold_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        primary_keys: Sequence[str],
        secondary_key: Optional[str] = None,
        attr_descriptions: Optional[Mapping[str, Mapping[str, Mapping]]] = None,
        query_type: Optional[str] = None,
    ) -> MatchResult:
        keys = list(primary_keys)
        if secondary_key and secondary_key not in keys:
            keys.append(secondary_key)

        self._ensure_columns_exist(gold_df, pred_df, keys)

        gold_norm = self._with_key_column(gold_df, keys)
        pred_norm = self._with_key_column(pred_df, keys)

        matched_gold_parts: List[pd.DataFrame] = []
        matched_pred_parts: List[pd.DataFrame] = []
        used_gold: set[int] = set()
        used_pred: set[int] = set()

        common_keys = sorted(set(gold_norm["__key"]) & set(pred_norm["__key"]))
        for key in common_keys:
            g_rows = gold_norm[gold_norm["__key"] == key]
            p_rows = pred_norm[pred_norm["__key"] == key]
            count = min(len(g_rows), len(p_rows))
            matched_gold_parts.append(g_rows.head(count).drop(columns="__key"))
            matched_pred_parts.append(p_rows.head(count).drop(columns="__key"))
            used_gold.update(g_rows.head(count).index)
            used_pred.update(p_rows.head(count).index)

        desc_map = self._build_description_map(attr_descriptions)
        key_context = self._build_key_context(keys, desc_map)
        for idx, g_row in gold_norm.iterrows():
            if idx in used_gold:
                continue
            best_pred = self._find_llm_match(
                g_row,
                pred_norm,
                keys,
                used_pred=used_pred,
                desc_map=desc_map,
                key_context=key_context,
                query_type=query_type,
            )
            if best_pred is not None:
                matched_gold_parts.append(g_row.to_frame().T.drop(columns="__key"))
                matched_pred_parts.append(pred_norm.loc[[best_pred]].drop(columns="__key"))
                used_gold.add(idx)
                used_pred.add(best_pred)

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

    def _build_description_map(self, attr_descriptions: Optional[Mapping[str, Mapping[str, Mapping]]]) -> Mapping[str, str]:
        """
        Flatten descriptions into a lowercase key -> description map.
        Supports both bare column names and table.column keys.
        """
        if not attr_descriptions:
            return {}
        desc_map: dict[str, str] = {}
        for table, cols in attr_descriptions.items():
            for col, meta in cols.items():
                desc = meta.get("description")
                if not desc:
                    continue
                desc_map[col.lower()] = desc
                desc_map[f"{table}.{col}".lower()] = desc
        return desc_map

    def _format_key_description(self, row: pd.Series, keys: Sequence[str], desc_map: Mapping[str, str]) -> str:
        parts: List[str] = []
        for key in keys:
            val = normalize_whitespace(row[key]) if key in row else ""
            desc = desc_map.get(key.lower())
            if desc:
                parts.append(f"{key} ({desc})={val}")
            else:
                parts.append(f"{key}={val}")
        return "; ".join(parts)

    def _build_key_context(self, keys: Sequence[str], desc_map: Mapping[str, str]) -> str:
        lines = []
        for key in keys:
            desc = desc_map.get(key.lower())
            if desc:
                lines.append(f"- {key}: {desc}")
        return "\n".join(lines)

    def _find_llm_match(
        self,
        gold_row: pd.Series,
        pred_df: pd.DataFrame,
        keys: Sequence[str],
        used_pred: set[int],
        desc_map: Mapping[str, str],
        key_context: str,
        query_type: Optional[str],
    ) -> Optional[int]:
        if not self.llm_client or not self.llm_client.can_use_llm:
            return None

        key_desc_gt = self._format_key_description(gold_row, keys, desc_map)
        for idx, pred_row in pred_df.iterrows():
            if idx in used_pred:
                continue
            key_desc_pred = self._format_key_description(pred_row, keys, desc_map)
            if self._llm_keys_match(key_desc_gt, key_desc_pred, key_context, query_type=query_type):
                return idx
        return None

    def _llm_keys_match(
        self, key_desc_gt: str, key_desc_pred: str, key_context: str, query_type: Optional[str]
    ) -> bool:
        if not self.llm_client or not self.llm_client.can_use_llm:
            return False

        messages = [
            {
                "role": "system",
                "content": (
                    "You judge whether two key descriptions refer to the same row/group in SQL results. "
                    "Respond ONLY YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query type: {query_type or 'select/filter'}\n"
                    f"Columns (with descriptions):\n{key_context or 'No additional description.'}\n\n"
                    f"Key A (ground truth):\n{key_desc_gt}\n\n"
                    f"Key B (prediction):\n{key_desc_pred}\n\n"
                    "Guidelines:\n"
                    "- Keys are sets of column=value pairs; all columns must match semantically.\n"
                    "- Allow synonyms, abbreviations, wording/casing differences if meaning is unchanged.\n"
                    "- Numeric values match if equal up to formatting/rounding.\n"
                    "- Missing/blank/unknown is NOT equal to a specific value.\n"
                    "- If any column conflicts, answer NO.\n\n"
                    "Return ONLY YES or NO."
                ),
            },
        ]
        verdict = self.llm_client._completion(messages, max_tokens=4)  # type: ignore[attr-defined]
        if not verdict:
            return False
        verdict = verdict.strip().upper()
        return verdict.startswith("Y") or verdict.startswith("T")
