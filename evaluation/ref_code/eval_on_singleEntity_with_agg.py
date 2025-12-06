#!/usr/bin/env python3
"""
单文档单实体（带聚合）结果评测脚本。
核心思路：
- 用同一份大 GT 表集合执行 SQL（可含 JOIN / GROUP BY / 聚合），得到该 SQL 的“小 GT”。
- 根据 GROUP BY 列作为主键对齐系统输出与 GT；必要时用 LLM 进行主键语义匹配。
- 逐列评估（聚合列/其他列沿用多实体脚本的语义匹配逻辑），输出 metrics.json 及对齐后的中间表。

路径尚未固定，请根据数据位置修改顶部大写变量：
- GT_TABLE_DIR: 存放所有 GT 表格的目录（每个表 1 个 CSV，文件名=表名）。
- SQL_DIR: SQL 或包含 {"sql": "...", "attr_description": {...}} 的 JSON 目录。
- RESULT_DIR: 系统输出 csv 的目录，文件名默认形如 sql_x_result.csv。
- OUTPUT_DIR: 评测输出（小 GT、排序结果、匹配结果、metrics.json）的根目录。
- RESULT_SUFFIX: 系统输出文件名去掉后缀后得到 SQL 同名前缀（默认 "_result"）。
- MODEL_NAME: litellm 使用的模型名，可按需调整。


/data/QUEST/jzshe/miniconda3/envs/quest/bin/python   evaluator/acc_cal/single_entity/eval_on_singleEntity_with_agg.py \
  --gt_dir UDA-Bench/Query/Player \
  --sql_dir UDA-Bench/Query/Player/Agg/sql_eval \
  --result_dir exp_result/player/agg_queries \
  --output_dir  exp_result/player/agg_queries/acc \
  --model_name openai/gpt-4.1-mini

  litellm-openrouter的模型使用：
  https://docs.litellm.ai/docs/providers/openrouter
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
from litellm import batch_completion, completion
from tqdm import tqdm

# ===== 可根据实际路径修改 ===== 直接在命令参数中指定也行。
GT_TABLE_DIR = "/data2/jproject/OpenQuestProject/UDA-Bench/Query/Player/player.csv"
SQL_DIR = "/path/to/sql_files"
RESULT_DIR = "/path/to/result_csv"
OUTPUT_DIR = "./outputs_single_entity_agg"
RESULT_SUFFIX = "_result"  # e.g., sql_1_result.csv -> sql_1.sql
MODEL_NAME = "openai/gpt-4.1-mini"
AGG_PREFIXES = ("avg_", "sum_", "min_", "max_", "count_")
# =================================

# 默认的 API 入口，可在运行前通过环境变量覆盖
########### aihub模型可用 ############
os.environ.setdefault("OPENAI_API_BASE", "https://aihubmix.com/v1")
os.environ.setdefault("OPENAI_API_KEY", "sk-fthhzHHMmUwA5cDq0eC213365c824c4f80B588C3E1557eB2")

# 尝试使用openrouter的免费模型

# os.environ["OPENROUTER_API_KEY"] = "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"
# os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

@dataclass(frozen=True)
class TableAlias:
    table: str
    alias: str


# ===================== 基础工具函数 =====================

def f1_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def safe_split(cell: Any, sep: str = "||") -> List[str]:
    """把单元格拆成列表，空值（包含 NULL/None/NaN）→ []，其余按分隔符拆并 strip。"""
    normalized = normalize_empty_cell(cell)
    if normalized == "":
        return []
    return [s.strip() for s in str(normalized).split(sep) if s.strip()]


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉 pandas 读 CSV 时带上的匿名索引列。"""
    keep_cols = [
        col for col in df.columns if str(col).strip() and not str(col).startswith("Unnamed")
    ]
    return df.loc[:, keep_cols]


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_string_whitespace(value: Any) -> Any:
    """去除首尾空白并压缩连续空白，非字符串原样返回。"""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return ""
        return WHITESPACE_PATTERN.sub(" ", stripped)
    return value


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅清洗字符串列，避免 group-by 键因额外空白导致错配。
    - 去掉首尾空白
    - 将连续空白规约为单个空格
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].apply(normalize_string_whitespace)
    return df


# ===================== 单元格归一化 =====================

EMPTY_LIKE_STRINGS = {"", "null", "none", "nan"}


def normalize_empty_cell(cell: Any) -> str:
    """
    将 '', 'NULL', 'None', 'NaN'、空白或 NaN 统统归一化为空字符串，大小写不敏感。
    其他值转成 strip 后的字符串返回。
    """
    if pd.isna(cell):
        return ""
    s = str(cell).strip()
    if s.lower() in EMPTY_LIKE_STRINGS:
        return ""
    return s


def normalize_key_cell(cell: Any) -> str:
    """
    主键列的专用归一化：在空值归一化基础上，把数字转成稳定的字符串形式，避免 0 与 0.0 之类的差异导致匹配失败。
    """
    base = normalize_empty_cell(cell)
    if base == "":
        return ""
    try:
        num = float(base)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return base


def normalize_cell_for_text(cell: Any) -> str:
    """
    在空值归一化基础上，尽量把数字转成统一格式的字符串，便于 LLM 判断。
    """
    base = normalize_empty_cell(cell)
    if base == "":
        return ""
    try:
        return str(float(base)).strip()
    except Exception:
        return base


def cells_equal(a: Any, b: Any) -> bool:
    """基于空值归一化后的精确比较。"""
    return normalize_empty_cell(a) == normalize_empty_cell(b)


# ===================== 数据加载 =====================

def load_gt_tables(gt_dir: str) -> Dict[str, pd.DataFrame]:
    """
    读取目录下的所有 CSV，表名取文件名（不含扩展名），全部注册为 duckdb 表。
    """
    gt_dir_path = Path(gt_dir)
    if not gt_dir_path.exists():
        raise FileNotFoundError(f"GT_TABLE_DIR 不存在：{gt_dir_path}")

    tables: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(gt_dir_path.glob("*.csv")):
        table_name = csv_path.stem
        df = pd.read_csv(csv_path)
        df = drop_unnamed_columns(df)
        df = normalize_string_columns(df)
        tables[table_name] = df

    if not tables:
        raise ValueError(f"{gt_dir} 下未找到任何 CSV。")

    return tables


def load_sql_payload(path: Path) -> Tuple[str, Dict[str, str]]:
    """
    支持两种格式：
    - .sql：文件内容即 SQL，attr_description 为空
    - .json：包含字段 sql 和 attr_description
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        sql = data.get("sql", "")
        if not sql:
            raise ValueError(f"{path} 缺少 sql 字段")
        attr_desc = data.get("attr_description", {}) or {}
        return sql, attr_desc

    sql = path.read_text()
    return sql, {}


# ===================== SQL 辅助 =====================

def extract_table_aliases(sql: str) -> List[TableAlias]:
    """
    从 FROM / JOIN 捕获表名与别名（如果未写别名，则 alias=table）。
    仅用于注册表名（兼容大小写/别名），不做 SQL 语法完整解析。
    """
    pattern = re.compile(
        r"(?i)\b(from|join)\s+([a-zA-Z_][\w]*)\s*(?:as\s+)?(?:([a-zA-Z_][\w]*))?"
    )
    aliases: List[TableAlias] = []
    seen: set[str] = set()
    for m in pattern.finditer(sql):
        table = m.group(2)
        alias = m.group(3) or table
        key = (table.lower(), alias.lower())
        if key in seen:
            continue
        seen.add(key)
        aliases.append(TableAlias(table=table, alias=alias))
    return aliases


def parse_group_by_columns(sql: str) -> List[str]:
    """
    从 SQL 中简单提取 GROUP BY 列表（不做完整解析，假定逗号分隔）。
    """
    m = re.search(r"(?is)\bgroup\s+by\s+(.+?)(?:\border\s+by\b|$)", sql)
    if not m:
        return []
    group_part = m.group(1)
    cols = []
    for part in group_part.split(","):
        col = part.strip().rstrip(";")
        # 去掉表别名前缀（形如 t.col）以及包裹的引号 / 反引号 / 空格
        m_alias = re.match(r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)$", col)
        if m_alias:
            col = m_alias.group(2)
        col = col.strip('"` ')
        if col:
            cols.append(col)
    return cols


# ===================== 行对齐（含 LLM 主键匹配） =====================

def format_key_description(
    row: pd.Series,
    key_cols: Sequence[str],
    attr_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    parts = []
    desc_map = attr_descriptions or {}
    for col in key_cols:
        val = normalize_empty_cell(row[col]) if col in row else ""
        desc = desc_map.get(col)
        desc_suffix = f" ({desc})" if desc else ""
        parts.append(f"{col}{desc_suffix}={val}")
    return "; ".join(parts)


def build_key_context(key_cols: Sequence[str], attr_descriptions: Optional[Dict[str, str]]) -> str:
    """
    将主键列的列描述拼接为上下文，便于 LLM 判断语义。
    """
    if not attr_descriptions:
        return ""
    lines = []
    for col in key_cols:
        desc = attr_descriptions.get(col)
        if desc:
            lines.append(f"- {col}: {desc}")
    return "\n".join(lines)


def llm_is_key_match(
    key_desc_gt: str,
    key_desc_pred: str,
    key_context: str = "",
    model_name: str = MODEL_NAME,
) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at judging whether two group-by key descriptions refer to the SAME bucket "
                "in aggregated SQL outputs. Return only 'YES' or 'NO'."
            ),
        },
        {
            "role": "user",
            "content": f"""Decide if the following two keys describe the same group in an aggregated SQL result.

Columns (with descriptions):
{key_context or "No additional description."}

Key A (ground truth):
{key_desc_gt}

Key B (prediction):
{key_desc_pred}

Guidelines:
- Each key is a set of column=value pairs. All columns must semantically match.
- Treat synonyms, abbreviations, spelling/wording variants, translations, adjective/demonym forms vs the underlying entity names, and casing/punctuation differences as equivalent when they mean the same thing.
- Allow reordered wording or minor typos if intent is clear.
- Numeric values match if they are equal up to formatting/rounding (e.g., 1, 1.0, 01).
- If one value is a more specific wording of the same entity (e.g., adds qualifiers like born in / naturalized / former) and introduces no conflicting entity, consider it equivalent.
- Composite values (e.g., A-B, A and B) only match if they describe the same combination, not just one part.
- Missing/blank/unknown values are NOT equivalent to specific values.
- If any column conflicts, answer NO.

Return ONLY YES or NO.""",
        },
    ]
    resp = completion(
        model=model_name,
        messages=messages,
        max_tokens=4,
        temperature=0,
    )
    content = resp["choices"][0]["message"]["content"].strip().upper()
    if content.startswith("Y") or content.startswith("T"):
        return True
    return False


def normalize_key_columns(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in key_cols:
        if col not in df.columns:
            raise KeyError(f"缺少关键列：{col}")
        df[col] = df[col].apply(normalize_key_cell)
    return df


def match_rows_with_llm(
    gt_df: pd.DataFrame,
    result_df: pd.DataFrame,
    key_cols: List[str],
    key_context: str = "",
    attr_descriptions: Optional[Dict[str, str]] = None,
    model_name: str = MODEL_NAME,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    先精确匹配 key tuple，再对未匹配的行用 LLM 判断 key 是否表示同一组。
    """
    gt_norm = normalize_key_columns(gt_df, key_cols)
    result_norm = normalize_key_columns(result_df, key_cols)

    matched_gt_rows = []
    matched_pred_rows = []

    pred_unused = list(result_norm.index)

    for _, gt_row in gt_norm.iterrows():
        gt_key = tuple(gt_row[col] for col in key_cols)

        exact_idx = None
        for j in pred_unused:
            pred_key = tuple(result_norm.loc[j, col] for col in key_cols)
            if pred_key == gt_key:
                exact_idx = j
                break

        if exact_idx is not None:
            matched_gt_rows.append(gt_row)
            matched_pred_rows.append(result_norm.loc[exact_idx])
            pred_unused.remove(exact_idx)
            continue

        key_desc_gt = format_key_description(gt_row, key_cols, attr_descriptions)
        llm_idx = None
        for j in list(pred_unused):
            pred_row = result_norm.loc[j]
            key_desc_pred = format_key_description(pred_row, key_cols, attr_descriptions)
            if llm_is_key_match(
                key_desc_gt, key_desc_pred, key_context=key_context, model_name=model_name
            ):
                llm_idx = j
                break

        if llm_idx is not None:
            matched_gt_rows.append(gt_row)
            matched_pred_rows.append(result_norm.loc[llm_idx])
            pred_unused.remove(llm_idx)

    if matched_gt_rows:
        matched_gt = pd.DataFrame(matched_gt_rows).reset_index(drop=True)
        matched_pred = pd.DataFrame(matched_pred_rows).reset_index(drop=True)
    else:
        matched_gt = gt_norm.head(0).copy()
        matched_pred = result_norm.head(0).copy()

    return matched_gt, matched_pred


def align_rows(
    gt_df: pd.DataFrame,
    result_df: pd.DataFrame,
    key_cols: List[str],
    attr_descriptions: Optional[Dict[str, str]] = None,
    model_name: str = MODEL_NAME,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    - 仅保留 GT 中出现的 key（避免多余预测行参与匹配）。
    - 精确 + LLM 对齐。
    - 返回：gt_sorted, result_sorted, matched_gt, matched_result, len_gt, len_pred
    """
    len_gt = len(gt_df)
    len_pred = len(result_df)

    # 先按 key 过滤预测行
    gt_norm = normalize_key_columns(gt_df, key_cols)
    result_norm = normalize_key_columns(result_df, key_cols)

    gt_keys = set(tuple(r[col] for col in key_cols) for _, r in gt_norm.iterrows())
    # 建议修复逻辑（不过滤可能语义匹配的 key）
    result_filtered = result_norm.copy()

    key_context = build_key_context(key_cols, attr_descriptions)
    matched_gt, matched_result = match_rows_with_llm(
        gt_norm,
        result_filtered,
        key_cols,
        key_context=key_context,
        attr_descriptions=attr_descriptions,
        model_name=model_name,
    )

    gt_sorted = gt_norm.sort_values(list(key_cols)).reset_index(drop=True)
    result_sorted = result_filtered.sort_values(list(key_cols)).reset_index(drop=True)

    return gt_sorted, result_sorted, matched_gt, matched_result, len_gt, len_pred


# ===================== 单元格评估（沿用多实体逻辑） =====================

def eval_unfixed_column_with_llm(
    col_pred,
    col_gt,
    length_pred: int,
    length_gt: int,
    column_name: str,
    column_description: Optional[str],
    sep_pred: str = "||",
    sep_gt: str = "||",
    batch_size: int = 8,
    model_name: str = MODEL_NAME,
) -> Tuple[float, float, float]:
    """
    在原有语义匹配逻辑上，增加列名与列描述的上下文。
    """
    col_pred = pd.Series(col_pred).reset_index(drop=True)
    col_gt = pd.Series(col_gt).reset_index(drop=True)

    if length_pred == 0 or length_gt == 0:
        return 0.0, 0.0, 0.0

    prompts = []
    len_pred_terms: List[int] = []
    len_gt_terms: List[int] = []

    for a, b in zip(col_pred, col_gt):
        str_a = normalize_cell_for_text(a)
        str_b = normalize_cell_for_text(b)

        a_split = safe_split(str_a, sep_pred)
        b_split = safe_split(str_b, sep_gt)

        len_pred_terms.append(len(a_split))
        len_gt_terms.append(len(b_split))

        desc_block = (
            f"Column: {column_name}\nDescription: {column_description}"
            if column_description
            else f"Column: {column_name}"
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an expert in judging whether predicted values semantically match ground truth values "
                    "for a specific table column. Respond strictly as instructed. Do not explain or add commentary."
                ),
            },
            {
                "role": "user",
                "content": f"""Use the column context to decide if each predicted term matches any ground-truth term.
{desc_block}

Predicted terms (List A): {a_split}
Ground truth terms (List B): {b_split}

Rules:
- Terms are separated by '||'. Treat each entry as an independent item.
- For each term in List A, determine if there exists ANY term in List B that conveys the same meaning (synonym, paraphrase, abbreviation, or equivalent numeric within 1% relative error).
- Count how many terms in List A have at least one semantic match in List B.

Return ONLY the final count as an integer. No explanation, no extra text.""",
            },
        ]
        prompts.append(prompt)

    results: List[int] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"LLM matching ({column_name})"):
        batch_prompts = prompts[i : i + batch_size]
        responses = batch_completion(
            model=model_name,
            messages=batch_prompts,
            stop=None,
            max_tokens=32,
            temperature=0,
        )
        for response in responses:
            content = response["choices"][0]["message"]["content"].strip()
            try:
                results.append(int(content))
            except Exception:
                results.append(0)

    row_count = len(results)
    precision_sum = 0.0
    recall_sum = 0.0

    for i in range(row_count):
        la = len_pred_terms[i]
        lb = len_gt_terms[i]
        match_cnt = results[i]

        if la == 0 and lb == 0:
            precision_i = 1.0
            recall_i = 1.0
        else:
            precision_i = (min(match_cnt, la) / la) if la != 0 else 0.0
            recall_i = (min(match_cnt, lb) / lb) if lb != 0 else 0.0

        precision_sum += precision_i
        recall_sum += recall_i

    precision = precision_sum / length_pred
    recall = recall_sum / length_gt
    f1 = f1_score(precision, recall)

    return precision, recall, f1


def eval_fixed_column(
    col_pred,
    col_gt,
    length_pred: int,
    length_gt: int,
) -> Tuple[float, float, float]:
    col_pred = pd.Series(col_pred).reset_index(drop=True)
    col_gt = pd.Series(col_gt).reset_index(drop=True)

    correct = 0
    for a, b in zip(col_pred, col_gt):
        if cells_equal(a, b):
            correct += 1

    if length_pred == 0 or length_gt == 0:
        return 0.0, 0.0, 0.0

    precision = correct / length_pred
    recall = correct / length_gt
    f1 = f1_score(precision, recall)
    return precision, recall, f1


def resolve_column_description(
    column_name: str, attr_descriptions: Optional[Dict[str, str]]
) -> Optional[str]:
    """
    优先使用精确匹配的列描述；若缺失且列名带聚合前缀（avg/sum/min/max/count），
    尝试剥去前缀用原列名匹配描述并补充说明。
    """
    if not attr_descriptions:
        return None

    direct = attr_descriptions.get(column_name)
    if direct:
        return direct

    lower_name = column_name.lower()

    for prefix in AGG_PREFIXES:
        if lower_name.startswith(prefix):
            base_key_lower = lower_name[len(prefix) :]
            base_entry = next(
                ((k, v) for k, v in attr_descriptions.items() if k.lower() == base_key_lower),
                None,
            )
            if base_entry:
                base_key, base_desc = base_entry
                return (
                    f"{column_name} aggregated from '{base_key}' (prefix '{prefix}'); "
                    f"base column description: {base_desc}"
                )

    # 兜底大小写不敏感匹配
    case_insensitive = next(
        (v for k, v in attr_descriptions.items() if k.lower() == lower_name), None
    )
    return case_insensitive


def evaluate_columns(
    matched_gt: pd.DataFrame,
    matched_pred: pd.DataFrame,
    length_gt: int,
    length_pred: int,
    key_columns: List[str],
    attr_descriptions: Optional[Dict[str, str]] = None,
    unfixed_columns: Optional[List[str]] = None,
    model_name: str = MODEL_NAME,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    common_cols = sorted(set(matched_gt.columns) & set(matched_pred.columns))

    exclude_cols = set(key_columns)
    eval_cols = [c for c in common_cols if c not in exclude_cols]

    if unfixed_columns is None or len(unfixed_columns) == 0:
        unfixed_set = set(eval_cols)
    else:
        unfixed_set = set(unfixed_columns)

    metrics_per_col: Dict[str, Dict[str, float]] = {}

    for col in tqdm(eval_cols, desc="Column accuracy"):
        col_gt = matched_gt[col]
        col_pred = matched_pred[col]
        col_desc = resolve_column_description(col, attr_descriptions)

        if col in unfixed_set:
            p, r, f1 = eval_unfixed_column_with_llm(
                col_pred,
                col_gt,
                length_pred=length_pred,
                length_gt=length_gt,
                column_name=col,
                column_description=col_desc,
                model_name=model_name,
            )
        else:
            p, r, f1 = eval_fixed_column(
                col_pred,
                col_gt,
                length_pred=length_pred,
                length_gt=length_gt,
            )

        metrics_per_col[col] = {"precision": p, "recall": r, "f1": f1}

    if len(metrics_per_col) == 0:
        avg_precision = avg_recall = avg_f1 = 0.0
    else:
        avg_precision = sum(m["precision"] for m in metrics_per_col.values()) / len(metrics_per_col)
        avg_recall = sum(m["recall"] for m in metrics_per_col.values()) / len(metrics_per_col)
        avg_f1 = sum(m["f1"] for m in metrics_per_col.values()) / len(metrics_per_col)

    avg_metrics = {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}
    return metrics_per_col, avg_metrics


# ===================== 主流程 =====================

def register_tables(con: duckdb.DuckDBPyConnection, tables: Dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        con.register(name, df)
        con.register(name.lower(), df)
        con.register(name.upper(), df)


def run_sql_on_gt(
    gt_tables: Dict[str, pd.DataFrame],
    sql: str,
) -> pd.DataFrame:
    con = duckdb.connect()
    register_tables(con, gt_tables)
    filtered_gt = con.execute(sql).df()
    con.close()

    return drop_unnamed_columns(filtered_gt)


def detect_key_columns_from_df(
    filtered_gt: pd.DataFrame,
    result_df: pd.DataFrame,
    group_by_cols: List[str],
) -> List[str]:
    """
    优先使用 GROUP BY 列名的交集；若为空，尝试 *.ID 或 ID；仍为空则使用所有公共列。
    """
    normalized_group_by = []
    for col in group_by_cols:
        col_clean = col.strip().rstrip(";").strip('"` ')
        m_alias = re.match(r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)$", col_clean)
        if m_alias:
            col_clean = m_alias.group(2)
        normalized_group_by.append(col_clean)

    keys = [c for c in normalized_group_by if c in filtered_gt.columns and c in result_df.columns]

    if not keys:
        dot_keys = sorted(
            [c for c in filtered_gt.columns if c.lower().endswith(".id") and c in result_df.columns]
        )
        if dot_keys:
            keys = dot_keys

    if not keys and "ID" in filtered_gt.columns and "ID" in result_df.columns:
        keys = ["ID"]

    if not keys:
        common_cols = sorted(set(filtered_gt.columns) & set(result_df.columns))
        if common_cols:
            keys = common_cols

    if not keys:
        raise ValueError("无法找到用于对齐的主键列，请检查 GROUP BY 列或结果列名。")

    return keys


def evaluate_single_sql_and_save(
    gt_tables: Dict[str, pd.DataFrame],
    sql_raw: str,
    result_df: pd.DataFrame,
    output_dir: Path,
    attr_descriptions: Optional[Dict[str, str]] = None,
    model_name: str = MODEL_NAME,
    unfixed_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    group_by_cols = parse_group_by_columns(sql_raw)

    filtered_gt = run_sql_on_gt(gt_tables, sql_raw)

    key_cols = detect_key_columns_from_df(filtered_gt, result_df, group_by_cols)
    gt_sorted, result_sorted, matched_gt, matched_result, len_gt, len_pred = align_rows(
        filtered_gt, result_df, key_cols, attr_descriptions=attr_descriptions, model_name=model_name
    )

    filtered_gt_path = output_dir / "sql_on_gt.csv"
    result_sorted_path = output_dir / "result_sorted.csv"
    matched_gt_path = output_dir / "matched_gt.csv"
    matched_result_path = output_dir / "matched_result.csv"

    filtered_gt.to_csv(filtered_gt_path, index=False, encoding="utf-8-sig")
    result_sorted.to_csv(result_sorted_path, index=False, encoding="utf-8-sig")
    matched_gt.to_csv(matched_gt_path, index=False, encoding="utf-8-sig")
    matched_result.to_csv(matched_result_path, index=False, encoding="utf-8-sig")

    metrics_per_col, avg_metrics = evaluate_columns(
        matched_gt,
        matched_result,
        length_gt=len_gt,
        length_pred=len_pred,
        key_columns=key_cols,
        attr_descriptions=attr_descriptions,
        unfixed_columns=unfixed_columns,
        model_name=model_name,
    )

    metrics_payload = {
        "acc_col": metrics_per_col,
        "acc_avg": avg_metrics,
        "len_gt": len_gt,
        "len_pred": len_pred,
        "key_columns": key_cols,
        "group_by": group_by_cols,
        "sql_used": sql_raw,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"[Info] SQL 已执行并写入: {filtered_gt_path}")
    print(f"[Info] 排序后结果已写入: {result_sorted_path}")
    print(f"[Info] 匹配行数: {len(matched_gt)} / GT 行数: {len_gt} / Pred 行数: {len_pred}")
    print(f"[Info] metrics.json 已写入: {metrics_path}")
    print(
        f"[Result] precision={avg_metrics['precision']:.4f}, "
        f"recall={avg_metrics['recall']:.4f}, f1={avg_metrics['f1']:.4f}"
    )

    return {
        "filtered_gt": filtered_gt,
        "result_sorted": result_sorted,
        "matched_gt": matched_gt,
        "matched_result": matched_result,
        "acc_col": metrics_per_col,
        "acc_avg": avg_metrics,
        "paths": {
            "filtered_gt": str(filtered_gt_path),
            "result_sorted": str(result_sorted_path),
            "matched_gt": str(matched_gt_path),
            "matched_result": str(matched_result_path),
            "metrics": str(metrics_path),
        },
    }


# ===================== CLI 与遍历文件 =====================

def parse_ids_filter(raw_ids: str) -> Optional[List[str]]:
    if not raw_ids:
        return None
    ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
    if not ids:
        return None
    return ids


def find_sql_file(sql_dir: Path, sql_base: str) -> Optional[Path]:
    candidates = [
        sql_dir / f"{sql_base}.json",
        sql_dir / f"{sql_base}.sql",
        sql_dir / f"{sql_base}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="单实体带聚合的 Extract-Filter-Join 结果评测")
    parser.add_argument("--gt_dir", type=str, default=GT_TABLE_DIR, help="GT 表目录，每表 1 个 CSV")
    parser.add_argument("--sql_dir", type=str, default=SQL_DIR, help="SQL 文件或 JSON 文件所在目录")
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR, help="系统输出 CSV 目录")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="评测结果输出目录")
    parser.add_argument("--result_suffix", type=str, default=RESULT_SUFFIX, help="结果文件名中的后缀，用于定位 SQL 前缀")
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="仅评测指定 id，逗号分隔，对应 sql_{id}_result.csv 的 {id} 部分。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="litellm 使用的模型名，例如 'openai/gpt-4.1-mini'",
    )
    parser.add_argument(
        "--unfixed_columns",
        type=str,
        default="",
        help="逗号分隔指定需要语义匹配的列名；为空则默认所有非主键列都用 LLM 语义匹配。",
    )

    args = parser.parse_args()

    ids_filter = parse_ids_filter(args.ids)
    unfixed_columns = parse_ids_filter(args.unfixed_columns) or None

    gt_tables = load_gt_tables(args.gt_dir)
    result_dir = Path(args.result_dir)
    sql_dir = Path(args.sql_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    result_files = sorted(result_dir.glob("*.csv"))
    if ids_filter is not None:
        ids_upper = {i.upper() for i in ids_filter}
        suffix_upper = args.result_suffix.upper()
        filtered = []
        for p in result_files:
            stem_upper = p.stem.upper()
            base = stem_upper[: -len(suffix_upper)] if stem_upper.endswith(suffix_upper) else stem_upper
            if base in ids_upper:
                filtered.append(p)
        result_files = filtered

    if not result_files:
        print("[Error] 未找到需要评测的 result CSV。")
        return

    print(f"[Info] GT 目录: {args.gt_dir}")
    print(f"[Info] 待评测文件数: {len(result_files)}")

    for csv_path in result_files:
        run_name = csv_path.stem
        suffix_upper = args.result_suffix.upper()
        stem_upper = run_name.upper()
        if stem_upper.endswith(suffix_upper):
            sql_base = run_name[: -len(args.result_suffix)]
        else:
            sql_base = run_name

        sql_path = find_sql_file(sql_dir, sql_base)
        if not sql_path:
            print(f"[Warn] 找不到对应 SQL 文件: {sql_base}，跳过 {run_name}")
            continue

        print(f"\n===== 评测 {run_name} =====")
        sql_raw, attr_description = load_sql_payload(sql_path)

        result_df = pd.read_csv(csv_path)
        result_df = drop_unnamed_columns(result_df)
        result_df = normalize_string_columns(result_df)

        run_output_dir = output_root / run_name
        evaluate_single_sql_and_save(
            gt_tables=gt_tables,
            sql_raw=sql_raw,
            result_df=result_df,
            output_dir=run_output_dir,
            attr_descriptions=attr_description,
            model_name=args.model_name,
            unfixed_columns=unfixed_columns,
        )


if __name__ == "__main__":
    main()
