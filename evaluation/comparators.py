from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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

    DEFAULT_API_BASE = "https://aihubmix.com/v1"
    DEFAULT_API_KEY = "sk-fthhzHHMmUwA5cDq0eC213365c824c4f80B588C3E1557eB2"
    DEFAULT_MODEL = "openai/gpt-4.1-mini"

    def __init__(self, settings: EvalSettings, logger_name: str = "llm_client") -> None:
        self.settings = settings
        self.logger = setup_logger(logger_name, level=settings.log_level)
        self._litellm = None
        self.model = settings.llm_model or self.DEFAULT_MODEL
        if settings.llm_provider and settings.llm_provider != "none":
            try:
                os.environ.setdefault("OPENAI_API_BASE", self.DEFAULT_API_BASE)
                os.environ.setdefault("OPENAI_API_KEY", self.DEFAULT_API_KEY)
                import litellm  # type: ignore

                self._litellm = litellm
            except Exception as exc:
                self.logger.warning("litellm not available, fallback to string match (%s)", exc)

    @property
    def can_use_llm(self) -> bool:
        return bool(self._litellm and self.model and self.settings.llm_provider != "none")

    def _completion(self, messages: List[dict], max_tokens: int = 8) -> Optional[str]:
        if not self.can_use_llm:
            return None
        try:
            resp = self._litellm.completion(  # type: ignore
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self.logger.warning("LLM completion failed, fallback to lexical compare: %s", exc)
            return None

    def compare(self, a: str, b: str, description: Optional[str] = None) -> bool:
        """
        For now, rely on normalized lexical comparison.
        Hook for plugging real LLM semantic matching if configured.
        """
        a_norm = normalize_whitespace(a)
        b_norm = normalize_whitespace(b)
        if a_norm.lower() == b_norm.lower():
            return True
        if not self.can_use_llm:
            return False

        desc_block = description or "No additional description."
        messages = [
            {
                "role": "system",
                "content": (
                    "You judge whether a predicted cell matches the ground truth for one column. "
                    "Respond with ONLY YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Column description: {desc_block}\n\n"
                    f"Predicted: {a_norm}\nGround truth: {b_norm}\n\n"
                    "Rules:\n"
                    "- Ignore casing, punctuation, and spacing differences.\n"
                    "- Synonyms, abbreviations, and obvious typos count as a match if meaning is the same.\n"
                    "- Numbers match if equal up to formatting/rounding (e.g., 1, 1.0, 01).\n"
                    "- Empty/unknown is NOT equal to a specific value.\n\n"
                    "Answer YES or NO."
                ),
            },
        ]
        content = self._completion(messages, max_tokens=4)
        if not content:
            return False
        verdict = content.strip().upper()
        return verdict.startswith("Y") or verdict.startswith("T")

    def match_term_count(
        self,
        pred_terms: Sequence[str],
        gold_terms: Sequence[str],
        column_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[int]:
        if not self.can_use_llm:
            return None

        desc_block = description or "No additional description."
        col_line = column_name or "N/A"
        messages = [
            {
                "role": "system",
                "content": (
                    "You count how many predicted terms have a semantic match in the ground truth list. "
                    "Return only the integer."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the column context to match terms.\n"
                    f"Column: {col_line}\n"
                    f"Description: {desc_block}\n\n"
                    f"Predicted terms (List A): {list(pred_terms)}\n"
                    f"Ground truth terms (List B): {list(gold_terms)}\n\n"
                    "Rules:\n"
                    "- Terms are separated by '||'; treat each as independent.\n"
                    "- A term matches if it is a synonym/paraphrase/abbreviation or a number within 1% relative error.\n"
                    "- Count how many terms in List A have at least one semantic match in List B.\n\n"
                    "Return ONLY the count as an integer."
                ),
            },
        ]
        content = self._completion(messages, max_tokens=12)
        if content is None:
            return None
        try:
            return int(content.strip())
        except Exception:
            self.logger.warning("Failed to parse LLM count result '%s', fallback to lexical match", content)
            return None


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

        llm_client = self.llm_comparator.llm_client
        matched: Optional[int] = llm_client.match_term_count(
            pred_values, gold_values, column_name=None, description=description
        )
        if matched is None:
            matched = self._lexical_match_count(pred_values, gold_values)

        precision = matched / len(pred_values) if pred_values else 0.0
        recall = matched / len(gold_values) if gold_values else 0.0
        return CellScore(precision=precision, recall=recall)

    def _lexical_match_count(self, pred_values: Sequence[str], gold_values: Sequence[str]) -> int:
        matched = 0
        used_pred = set()
        for g in gold_values:
            g_norm = normalize_whitespace(g).lower()
            for idx, p in enumerate(pred_values):
                if idx in used_pred:
                    continue
                p_norm = normalize_whitespace(p).lower()
                if p_norm == g_norm:
                    matched += 1
                    used_pred.add(idx)
                    break
        return matched
