# typing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable, Any, TypedDict

PKType = List[str]              # 支持多主键
Delimiter = "||"              # 多值单元格分隔符，如 "||"

class EvalDict(TypedDict):
    overall: Dict[str, float]
    by_column: Dict[str, Dict[str, float]]

@dataclass
class PKSchema:
    keys: PKType                 # 主键列名
    fuzzy: bool = False          # 是否启用模糊匹配，只有group by的列有可能作为主键参与匹配

