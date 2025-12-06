#!/usr/bin/env python3
"""
基于 medicine-quest 数据的多实体抽取结果评测脚本。
相关数据集见： /data2/jproject/questExperiment/experiment_result/acc_multientity_test
支持匹配gt和result中的空值(对应的cell都为空会被判定为正确)

使用示例：
python eval_on_medicine.py \
  --gt_csv /data2/jproject/questExperiment/experiment_result/acc_multientity_test/medicine-quest/gt/medical.csv \
  --result_dir /data2/jproject/questExperiment/experiment_result/acc_multientity_test/medicine-quest/result \
  --sql_dir /data2/jproject/questExperiment/experiment_result/acc_multientity_test/medicine-quest/sql_with_description \
  --output_dir ./outputs_medicine \
  --model_name openai/gpt-4.1-mini

已有的相关格式要求：
gt_csv: 第一行是表头，其中一定要包含1个ID列，实际的ID值是第一个下划线前面的数值
result_dir: 每个 CSV 文件的第一行是表头，其中一定要包含1个file_name列，去除文件名后缀后能变成ID值
sql_dir: 每个 JSON 文件包含 sql 和 attr_description 字段，sql 用于过滤 GT 表格，attr_description 用于提供列描述
```
{
    "sql": "SELECT disease_name, preventive_measures, common_symptoms, diagnosis_challenges FROM medical",
    "attr_description": {
        "disease_name": "the official name of the disease (e.g., Hypertension).",
        "preventive_measures": "known methods or actions to reduce the risk of disease occurrence, choose one or more from ['vaccination', 'lifestyle_modification', 'screening', 'prophylactic_medication', 'personal_hygiene', 'health_education', 'protective_equipment', 'environmental_control', 'vector_control', 'safe_sex_practices', 'regular_exercise', 'dietary_management', 'early_detection', 'other'], , seperated by `||` (e.g. vaccination || regular_exercise).",
        "common_symptoms": "typical clinical manifestations or symptoms experienced by patients with this disease (e.g., fever, cough, joint pain).",
        "diagnosis_challenges": "typical difficulties or obstacles encountered in the diagnosis of the disease (e.g., nonspecific symptoms, lack of early biomarkers, disease mimics)."
    }
}

```

说明：
- result 目录下的文件名是 SF{id}.csv 或 SFW{id}.csv。
- 同名（转成小写）的 sql_with_description/sf{id}.json|sfw{id}.json 存放了 SQL 以及各列的描述。
- 某些 result CSV 的首列可能是 pandas 自动带的 index（列名为空或以 Unnamed 开头），会自动丢弃，不参与计算。
- 评测时会把列的 description 加入到 LLM prompt 中以提高判断可靠性。
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from litellm import batch_completion, completion
from tqdm import tqdm

# 统一设置 API 入口
os.environ["OPENAI_API_BASE"] = "https://aihubmix.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-fthhzHHMmUwA5cDq0eC213365c824c4f80B588C3E1557eB2"

DEFAULT_GT_CSV = "/data2/jproject/questExperiment/experiment_result/acc_multientity_test/medicine-quest/gt/medical.csv"
DEFAULT_RESULT_DIR = "/data2/jproject/questExperiment/experiment_result/acc_multientity_test/medicine-quest/result"
DEFAULT_SQL_DIR = "/data2/jproject/questExperiment/experiment_result/acc_multientity_test/medicine-quest/sql_with_description"
DEFAULT_OUTPUT_DIR = "./outputs_medicine"


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
    仅清洗字符串列，避免额外空白符导致的错配。
    - 去掉首尾空白符
    - 连续空白规约为单个空格
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


# ===================== SQL 处理 =====================

def ensure_from_medical(sql: str, table_name: str = "medical") -> str:
    """保证 SQL 里有 FROM medical。"""
    if re.search(rf"(?i)\bfrom\s+{table_name}\b", sql):
        return sql

    where_pattern = re.compile(r"(?i)\bwhere\b")
    if where_pattern.search(sql):
        return where_pattern.sub(f"FROM {table_name} WHERE", sql, count=1)

    return sql.rstrip().rstrip(";") + f" FROM {table_name}"


def ensure_id_selected(sql: str) -> str:
    """
    确保 SELECT 列表里包含 id。
    简单启发式：在开头 SELECT（或 SELECT DISTINCT）后面直接插入 'id, '。
    """
    if re.search(r"(?i)\bID\b", sql):
        return sql

    select_pattern = re.compile(r"(?i)^\s*select\s+(distinct\s+)?")
    m = select_pattern.search(sql)
    if m:
        insert_pos = m.end()
        return sql[:insert_pos] + "id, " + sql[insert_pos:]

    # 极端情况：未匹配到 SELECT，直接前置 id, 保持原 SQL 结构
    return f"SELECT id, {sql.lstrip()}"


def normalize_sql(sql: str, table_name: str = "medical") -> str:
    sql_with_from = ensure_from_medical(sql, table_name=table_name)
    return ensure_id_selected(sql_with_from)


def load_sql_and_description(json_path: Path) -> Tuple[str, Dict[str, str]]:
    data = json.loads(json_path.read_text())
    sql = data.get("sql", "")
    attr_description = data.get("attr_description", {})
    if not sql:
        raise ValueError(f"{json_path} 缺少 sql 字段")
    if not attr_description:
        raise ValueError(f"{json_path} 缺少 attr_description 字段")
    return sql, attr_description


# ===================== unfixed 列的 LLM 评估 =====================

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
    model_name: str = "openai/gpt-4.1-mini",
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
            f"Column: {column_name}\\nDescription: {column_description}"
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


# ===================== fixed 列（精确匹配）的评估 =====================

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


# ===================== primary_key 的 LLM 匹配 =====================

def llm_is_primary_match(
    gt_value: str,
    pred_value: str,
    model_name: str = "openai/gpt-4.1-mini",
) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at judging whether two mentions refer to the SAME real-world entity. "
                "Return only 'YES' or 'NO'."
            ),
        },
        {
            "role": "user",
            "content": f"""We have two mentions of a possible entity.

Mention A: {gt_value}
Mention B: {pred_value}

If they clearly refer to the same entity (even with abbreviations, typos, or synonyms), answer YES.
Otherwise answer NO.

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


def primary_match_with_llm(
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    primary_key: str,
    model_name: str = "openai/gpt-4.1-mini",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    matched_gt_rows = []
    matched_pred_rows = []

    gt_file_ids = df_gt["file_id"].apply(normalize_empty_cell)
    pred_file_ids = df_pred["file_id"].apply(normalize_empty_cell)

    common_file_ids = sorted(set(gt_file_ids) & set(pred_file_ids))
    common_file_ids = [fid for fid in common_file_ids if fid != ""]

    total_gt_rows = len(df_gt[df_gt["file_id"].isin(common_file_ids)])
    progress = tqdm(total=total_gt_rows, desc="Primary match by row")

    try:
        for fid in common_file_ids:
            gt_indices = gt_file_ids[gt_file_ids == fid].index
            pred_indices = pred_file_ids[pred_file_ids == fid].index

            gt_group = df_gt.loc[gt_indices]
            pred_group = df_pred.loc[pred_indices]

            unused_pred_indices = list(pred_group.index)

            for _, gt_row in gt_group.iterrows():
                try:
                    gt_pk = normalize_empty_cell(gt_row[primary_key])

                    exact_idx = None
                    for j in unused_pred_indices:
                        pred_pk = normalize_empty_cell(pred_group.loc[j, primary_key])
                        if pred_pk == gt_pk:
                            exact_idx = j
                            break

                    if exact_idx is not None:
                        matched_gt_rows.append(gt_row)
                        matched_pred_rows.append(pred_group.loc[exact_idx])
                        unused_pred_indices.remove(exact_idx)
                        continue

                    if gt_pk == "":
                        # 空 primary_key 只能与同为空的行匹配，已在精确匹配处理过
                        continue

                    llm_idx = None
                    for j in list(unused_pred_indices):
                        pred_pk = normalize_empty_cell(pred_group.loc[j, primary_key])
                        if pred_pk == "":
                            continue
                        if llm_is_primary_match(gt_pk, pred_pk, model_name=model_name):
                            llm_idx = j
                            break

                    if llm_idx is not None:
                        matched_gt_rows.append(gt_row)
                        matched_pred_rows.append(pred_group.loc[llm_idx])
                        unused_pred_indices.remove(llm_idx)
                        continue
                finally:
                    progress.update(1)
    finally:
        progress.close()

    if len(matched_gt_rows) == 0:
        matched_gt = df_gt.head(0).copy()
        matched_pred = df_pred.head(0).copy()
    else:
        matched_gt = pd.DataFrame(matched_gt_rows).reset_index(drop=True)
        matched_pred = pd.DataFrame(matched_pred_rows).reset_index(drop=True)

    return matched_gt, matched_pred


# ===================== 逐列评估 & 汇总 =====================

def evaluate_columns(
    matched_gt: pd.DataFrame,
    matched_pred: pd.DataFrame,
    length_gt: int,
    length_pred: int,
    primary_key: str,
    attr_descriptions: Optional[Dict[str, str]] = None,
    unfixed_columns: Optional[List[str]] = None,
    model_name: str = "openai/gpt-4.1-mini",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    common_cols = sorted(set(matched_gt.columns) & set(matched_pred.columns))

    exclude_cols = {"id", "file_name", "file_id"}
    eval_cols = [c for c in common_cols if c not in exclude_cols]

    if unfixed_columns is None or len(unfixed_columns) == 0:
        unfixed_set = set(eval_cols)
    else:
        unfixed_set = set(unfixed_columns)

    desc_map = attr_descriptions or {}
    metrics_per_col: Dict[str, Dict[str, float]] = {}

    for col in tqdm(eval_cols, desc="Column accuracy"):
        col_gt = matched_gt[col]
        col_pred = matched_pred[col]
        col_desc = desc_map.get(col)

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


# ===================== 主流程：读取文件 + duckdb 过滤 + 匹配 + 评估 =====================

def run_sql_on_gt(
    gt_df: pd.DataFrame,
    sql: str,
    table_name: str = "medical",
) -> pd.DataFrame:
    con = duckdb.connect()
    con.register(table_name, gt_df)
    filtered_gt = con.execute(sql).df()
    con.close()

    if "id" not in filtered_gt.columns:
        raise ValueError("SQL 结果中必须包含 'id' 列，请检查 SQL 构造逻辑。")

    return filtered_gt


def add_file_id_columns(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_gt = df_gt.copy()
    df_pred = df_pred.copy()

    df_gt["id"] = df_gt["id"].apply(normalize_empty_cell)
    df_gt["file_id"] = df_gt["id"].str.split("_", n=1, expand=True)[0]

    df_pred["file_name"] = df_pred["file_name"].apply(normalize_empty_cell)
    df_pred["file_id"] = df_pred["file_name"].str.replace(".txt", "", regex=False)

    return df_gt, df_pred


def evaluate_single_sql_and_save(
    gt_df: pd.DataFrame,
    result_df: pd.DataFrame,
    sql: str,
    primary_key: str,
    attr_descriptions: Optional[Dict[str, str]] = None,
    unfixed_columns: Optional[List[str]] = None,
    model_name: str = "openai/gpt-4.1-mini",
    output_dir: str = "./outputs",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n================ SQL ================\n{sql}\n")

    filtered_gt = run_sql_on_gt(gt_df, sql, table_name="medical")

    filtered_gt_path = os.path.join(output_dir, "filtered_gt.csv")
    filtered_gt.to_csv(filtered_gt_path, index=False, encoding="utf-8-sig")
    print(f"[Info] filtered_gt 已保存到: {filtered_gt_path}")
    print(f"[Info] filtered_gt 行数: {len(filtered_gt)}")

    filtered_gt_with_fid, result_with_fid = add_file_id_columns(filtered_gt, result_df)

    target_file_ids = set(filtered_gt_with_fid["file_id"].astype(str))
    result_sub = result_with_fid[result_with_fid["file_id"].astype(str).isin(target_file_ids)].copy()

    length_gt = len(filtered_gt_with_fid)
    length_pred = len(result_sub)

    print(f"[Info] 相关 prediction 行数: {length_pred}")

    print("[Step] 开始 primary_match（精确 + LLM）...")
    matched_gt, matched_result = primary_match_with_llm(
        filtered_gt_with_fid,
        result_sub,
        primary_key=primary_key,
        model_name=model_name,
    )
    print(f"[Info] 匹配成功的行数: {len(matched_gt)}")

    matched_gt_path = os.path.join(output_dir, "matched_gt.csv")
    matched_result_path = os.path.join(output_dir, "matched_result.csv")
    matched_gt.to_csv(matched_gt_path, index=False, encoding="utf-8-sig")
    matched_result.to_csv(matched_result_path, index=False, encoding="utf-8-sig")
    print(f"[Info] matched_gt 已保存到: {matched_gt_path}")
    print(f"[Info] matched_result 已保存到: {matched_result_path}")

    print("[Step] 开始列级别评估 (col_acc_match)...")
    metrics_per_col, avg_metrics = evaluate_columns(
        matched_gt,
        matched_result,
        length_gt=length_gt,
        length_pred=length_pred,
        primary_key=primary_key,
        attr_descriptions=attr_descriptions,
        unfixed_columns=unfixed_columns,
        model_name=model_name,
    )

    print("\n[Result] 每列指标：")
    for col, m in metrics_per_col.items():
        print(
            f"  - {col}: precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, f1={m['f1']:.4f}"
        )

    print("\n[Result] 平均指标（所有列的简单平均）：")
    print(
        f"  precision={avg_metrics['precision']:.4f}, "
        f"recall={avg_metrics['recall']:.4f}, f1={avg_metrics['f1']:.4f}"
    )

    metrics_payload = {"acc_col": metrics_per_col, "acc_avg": avg_metrics}
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"[Info] acc_col / acc_avg 已保存到: {metrics_path}")

    return {
        "filtered_gt": filtered_gt,
        "matched_gt": matched_gt,
        "matched_result": matched_result,
        "acc_col": metrics_per_col,
        "acc_avg": avg_metrics,
        "paths": {
            "filtered_gt": filtered_gt_path,
            "matched_gt": matched_gt_path,
            "matched_result": matched_result_path,
            "metrics": metrics_path,
        },
    }


def parse_ids_filter(raw_ids: str) -> Optional[List[str]]:
    if not raw_ids:
        return None
    ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
    if not ids:
        return None
    return ids


def extract_first_select_column(sql: str) -> Optional[str]:
    """
    简单从原始 SQL 的 SELECT 列表里取第一个字段（忽略 DISTINCT）。
    仅做轻量级字符串切分，不处理嵌套/函数等复杂情况。
    """
    m = re.search(r"(?is)select\s+(distinct\s+)?(.+?)\s+from\b", sql)
    if not m:
        return None
    select_part = m.group(2).strip()
    if not select_part:
        return None
    first = select_part.split(",")[0].strip()
    # 去掉 AS/别名与引号
    first = re.split(r"(?i)\s+as\s+", first)[0].strip()
    first = first.strip("`\"' ")
    return first or None


def determine_primary_key(sql_raw: str, override: Optional[str]) -> str:
    """
    优先使用命令行 override；否则取原始 SQL（补 id 之前）里 SELECT 的第一个字段。
    """
    if override:
        return override

    first_col = extract_first_select_column(sql_raw)
    if first_col:
        return first_col

    raise ValueError("无法确定 primary_key：请提供 --primary_key，或确保 SQL 的 SELECT 中包含至少一个字段。")


def main():
    parser = argparse.ArgumentParser(description="medicine-quest 结果评测脚本")
    parser.add_argument("--gt_csv", type=str, default=DEFAULT_GT_CSV, help="GT 表格 CSV 路径")
    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR, help="结果 CSV 目录")
    parser.add_argument("--sql_dir", type=str, default=DEFAULT_SQL_DIR, help="sql_with_description 目录")
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="只评测指定的文件名（不含扩展名），逗号分隔，如 'SF1,SFW3'。不填则评测目录下全部。",
    )
    parser.add_argument(
        "--primary_key",
        type=str,
        default="",
        help="手动指定 primary_key。若不指定，默认取对应 JSON 中 attr_description 的第一个键。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-4.1-mini",
        help="litellm 使用的模型名，例如 'openai/gpt-4.1-mini' 或 'openai/openai/gpt-4.1-mini'",
    )
    parser.add_argument(
        "--output_dir",
        "--save_dir",
        dest="output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="保存 filtered_gt / matched_gt / matched_result / metrics.json 的根目录",
    )

    args = parser.parse_args()

    ids_filter = parse_ids_filter(args.ids)

    gt_df = pd.read_csv(args.gt_csv)
    gt_df = drop_unnamed_columns(gt_df)
    gt_df = normalize_string_columns(gt_df)

    result_dir = Path(args.result_dir)
    sql_dir = Path(args.sql_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    result_files = sorted(result_dir.glob("*.csv"))
    if ids_filter is not None:
        ids_upper = {i.upper() for i in ids_filter}
        result_files = [p for p in result_files if p.stem.upper() in ids_upper]

    if not result_files:
        print("[Error] 未找到需要评测的 result CSV。")
        return

    print(f"[Info] GT 路径: {args.gt_csv}")
    print(f"[Info] 待评测文件数: {len(result_files)}")

    for csv_path in result_files:
        run_name = csv_path.stem  # e.g., SF1
        json_name = run_name.lower() + ".json"
        json_path = sql_dir / json_name
        if not json_path.exists():
            print(f"[Warn] 找不到对应的 SQL 描述文件: {json_path}，跳过 {run_name}")
            continue

        print(f"\n===== 评测 {run_name} =====")
        sql_raw, attr_description = load_sql_and_description(json_path)
        sql = normalize_sql(sql_raw, table_name="medical")
        primary_key = determine_primary_key(sql_raw, override=args.primary_key.strip() or None)

        result_df = pd.read_csv(csv_path)
        result_df = drop_unnamed_columns(result_df)
        result_df = normalize_string_columns(result_df)

        run_output_dir = output_root / run_name

        evaluate_single_sql_and_save(
            gt_df=gt_df,
            result_df=result_df,
            sql=sql,
            primary_key=primary_key,
            attr_descriptions=attr_description,
            model_name=args.model_name,
            output_dir=str(run_output_dir),
        )


if __name__ == "__main__":
    main()
