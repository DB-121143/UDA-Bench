"""
Modular evaluation toolkit for UDA-Bench.

Modules are organized to match the design in ``evaluation/doc/modular_eval_script_design.md``.
Each piece can be imported independently or wired together via ``run_eval.py``.
"""

__all__ = [
    "config",
    "sql_parser",
    "query_manifest",
    "sql_preprocessor",
    "gt_runner",
    "result_loader",
    "row_matcher",
    "comparators",
    "metrics",
    "result_writer",
]
