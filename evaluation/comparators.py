from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .config import EvalSettings
from .logging_utils import setup_logger
from .utils import normalize_whitespace, split_multi_value


def f1_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


@dataclass
class CellScore:
    precision: float
    recall: float

    @property
    def f1(self) -> float:
        return f1_score(self.precision, self.recall)


class LlmClient:
    """
    Minimal LLM wrapper.
    Falls back to deterministic comparison when provider is disabled or unavailable.
    """

    def __init__(self, settings: EvalSettings, logger_name: str = "llm_client") -> None:
        self.settings = settings
        self.logger = setup_logger(logger_name, level=settings.log_level)
        self._litellm = None
        if settings.llm_provider and settings.llm_provider != "none":
            try:
                import litellm  # type: ignore

                self._litellm = litellm
            except Exception as exc:
                self.logger.warning("litellm not available, fallback to string match (%s)", exc)

    def compare(self, a: str, b: str, description: Optional[str] = None) -> bool:
        """
        For now, rely on normalized lexical comparison.
        Hook for plugging real LLM semantic matching if configured.
        """
        a_norm = normalize_whitespace(a).lower()
        b_norm = normalize_whitespace(b).lower()
        if a_norm == b_norm:
            return True
        # Placeholder: if LLM available, could call for semantic match.
        return False


class CellComparator:
    def compare(self, pred: str, gold: str, description: Optional[str] = None) -> CellScore:
        raise NotImplementedError


class NumericComparator(CellComparator):
    def __init__(self, settings: EvalSettings) -> None:
        self.settings = settings

    def compare(self, pred, gold, description: Optional[str] = None) -> CellScore:
        try:
            if isinstance(pred, str):
                pred_val = float(pred.strip())
            else:
                pred_val = float(pred)
            if isinstance(gold, str):
                gold_val = float(gold.strip())
            else:
                gold_val = float(gold)
        except Exception:
            return CellScore(precision=0.0, recall=0.0)

        if math.isnan(pred_val) or math.isnan(gold_val):
            return CellScore(precision=0.0, recall=0.0)

        if self.settings.float_tolerance and not float(self.settings.float_tolerance) == 0:
            ok = abs(pred_val - gold_val) <= self.settings.float_tolerance
        else:
            ok = pred_val == gold_val

        score = 1.0 if ok else 0.0
        return CellScore(precision=score, recall=score)


class AggComparator(CellComparator):
    def compare(self, pred, gold, description: Optional[str] = None) -> CellScore:
        try:
            pred_val = float(pred)
            gold_val = float(gold)
        except Exception:
            return CellScore(precision=0.0, recall=0.0)
        if gold_val == 0:
            ok = pred_val == gold_val
            score = 1.0 if ok else 0.0
            return CellScore(precision=score, recall=score)
        relative_error = abs(pred_val - gold_val) / abs(gold_val)
        score = 1 / (1 + relative_error)
        return CellScore(precision=score, recall=score)


class StringLLMComparator(CellComparator):
    def __init__(self, settings: EvalSettings, llm_client: Optional[LlmClient] = None) -> None:
        self.settings = settings
        self.llm_client = llm_client or LlmClient(settings)

    def compare(self, pred, gold, description: Optional[str] = None) -> CellScore:
        pred_norm = normalize_whitespace(pred)
        gold_norm = normalize_whitespace(gold)
        if pred_norm == "" and gold_norm == "":
            return CellScore(precision=1.0, recall=1.0)
        equal = self.llm_client.compare(pred_norm, gold_norm, description=description)
        score = 1.0 if equal else 0.0
        return CellScore(precision=score, recall=score)


class MultiValueComparator(CellComparator):
    def __init__(self, settings: EvalSettings, llm_comparator: Optional[StringLLMComparator] = None) -> None:
        self.settings = settings
        self.llm_comparator = llm_comparator or StringLLMComparator(settings)

    def compare(self, pred, gold, description: Optional[str] = None) -> CellScore:
        pred_values = split_multi_value(pred, sep=self.settings.multi_value_sep)
        gold_values = split_multi_value(gold, sep=self.settings.multi_value_sep)

        if not pred_values and not gold_values:
            return CellScore(precision=1.0, recall=1.0)
        if not gold_values:
            return CellScore(precision=0.0, recall=0.0)

        matched = 0
        used_pred = set()
        for g in gold_values:
            for idx, p in enumerate(pred_values):
                if idx in used_pred:
                    continue
                if self.llm_comparator.llm_client.compare(p, g, description=description):
                    matched += 1
                    used_pred.add(idx)
                    break

        precision = matched / len(pred_values) if pred_values else 0.0
        recall = matched / len(gold_values) if gold_values else 0.0
        return CellScore(precision=precision, recall=recall)
