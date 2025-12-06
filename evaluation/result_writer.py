from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import dump_json
from .logging_utils import setup_logger
from .utils import ensure_dir


class ResultWriter:
    def __init__(self, output_dir: Path, logger_name: str = "result_writer") -> None:
        self.output_dir = Path(output_dir)
        self.logger = setup_logger(logger_name)
        ensure_dir(self.output_dir)

    def write(
        self,
        gold_df: pd.DataFrame,
        matched_gold_df: pd.DataFrame,
        matched_pred_df: pd.DataFrame,
        metrics: Dict,
    ) -> None:
        gold_path = self.output_dir / "gold_result.csv"
        matched_gold_path = self.output_dir / "matched_gold_result.csv"
        matched_pred_path = self.output_dir / "matched_result.csv"
        acc_path = self.output_dir / "acc.json"

        gold_df.to_csv(gold_path, index=False)
        matched_gold_df.to_csv(matched_gold_path, index=False)
        matched_pred_df.to_csv(matched_pred_path, index=False)
        dump_json(metrics, acc_path)

        self.logger.info(
            "Saved outputs to %s (gold=%s, matched=%s rows)",
            self.output_dir,
            len(gold_df),
            len(matched_gold_df),
        )
