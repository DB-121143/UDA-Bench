from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .comparators import (
    AggComparator,
    CellComparator,
    MultiValueComparator,
    NumericComparator,
    StringLLMComparator,
    f1_score,
)
from .config import EvalSettings
from .logging_utils import setup_logger
from .query_manifest import QueryManifest
from .row_matcher import MatchResult


class MetricCalculator:
    """Compute per-column and macro metrics."""

    def __init__(self, manifest: QueryManifest, settings: EvalSettings) -> None:
        self.manifest = manifest
        self.settings = settings
        self.logger = setup_logger("metric_calculator", level=settings.log_level)
        self.numeric = NumericComparator(settings)
        self.string = StringLLMComparator(settings)
        self.multi = MultiValueComparator(settings)
        self.agg = AggComparator()

    def compute(self, match_result: MatchResult) -> Dict:
        gold_df = match_result.gold_aligned
        pred_df = match_result.pred_aligned

        column_metrics: Dict[str, Dict[str, float]] = {}
        evaluated_columns: List[str] = []

        for item in self.manifest.parsed.select_items:
            col_name = item.output_name
            if col_name in self.manifest.stop_columns:
                continue
            if col_name not in gold_df.columns or col_name not in pred_df.columns:
                self.logger.warning("Skipping column %s because it is missing in aligned data", col_name)
                continue
            comparator = self._pick_comparator(item)
            meta = self.manifest.get_column_meta(col_name)
            precision_sum = 0.0
            recall_sum = 0.0
            for pred_cell, gold_cell in zip(pred_df[col_name], gold_df[col_name]):
                score = comparator.compare(pred_cell, gold_cell, description=meta.description if meta else None)
                precision_sum += score.precision
                recall_sum += score.recall

            precision = precision_sum / match_result.len_pred if match_result.len_pred else 0.0
            recall = recall_sum / match_result.len_gold if match_result.len_gold else 0.0
            f1 = f1_score(precision, recall)
            column_metrics[col_name] = {"precision": precision, "recall": recall, "f1": f1}
            evaluated_columns.append(col_name)

        macro = self._macro_average(column_metrics)

        return {
            "columns": column_metrics,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
            "rows": {
                "len_gold": match_result.len_gold,
                "len_pred": match_result.len_pred,
                "matched_rows": match_result.matched_rows,
            },
            "warnings": match_result.warnings,
            "used_keys": match_result.keys,
            "evaluated_columns": evaluated_columns,
            "skipped_columns": [c for c in self.manifest.parsed.output_columns if c not in evaluated_columns],
        }

    def _pick_comparator(self, item) -> CellComparator:
        meta = self.manifest.get_column_meta(item.output_name)
        if item.is_agg:
            return self.agg
        if meta and meta.value_type in {"int", "float"}:
            return self.numeric
        if meta and meta.value_type == "multi-str":
            return self.multi
        return self.string

    def _macro_average(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not metrics:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        p = sum(m["precision"] for m in metrics.values()) / len(metrics)
        r = sum(m["recall"] for m in metrics.values()) / len(metrics)
        f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
        return {"precision": p, "recall": r, "f1": f1}
