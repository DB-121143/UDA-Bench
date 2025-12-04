
import os
import sys
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import re

from tqdm import tqdm
os.environ["OPENAI_API_BASE"] = "https://aihubmix.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b"

from quest.core.nlp.match.table_matcher.table_join import create_advanced_group_by_matcher


import random

def drop_duplicate_primary_keys_rows(df: pd.DataFrame, primary_keys: list) -> pd.DataFrame:
    """
    对于主键列存在重复的行，随机保留一行，其余全部删除。
    返回去重后的DataFrame。
    """
    # 生成主键组合列
    if len(primary_keys) == 1:
        key_col = primary_keys[0]
        # groupby主键，随机采样每组一行
        dedup_df = df.groupby(key_col, group_keys=False).apply(lambda x: x.sample(1, random_state=random.randint(0, 99999)))
    else:
        # 多主键，先转为tuple
        key_tuples = df[primary_keys].astype(str).apply(tuple, axis=1)
        df = df.copy()
        df['__merge_key__'] = key_tuples
        dedup_df = df.groupby('__merge_key__', group_keys=False).apply(lambda x: x.sample(1, random_state=random.randint(0, 99999)))
        dedup_df = dedup_df.drop(columns='__merge_key__')
    dedup_df = dedup_df.reset_index(drop=True)
    return dedup_df

def primary_key_intersect(primary_keys: List[str], gt: pd.DataFrame, pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据主键类型，返回gt和pred中主键匹配的子集（已对齐顺序）。
    - id: 精确匹配
    - 多主键: 精确匹配
    - 单主键且不是id: fuse_match模糊匹配
    """
    if len(primary_keys) == 1:
        key = primary_keys[0]
        if key == "id":
            # 精确匹配
            keys_inter = set(gt[key]) & set(pred[key])
            gt_sub = gt[gt[key].isin(keys_inter)].copy()
            pred_sub = pred[pred[key].isin(keys_inter)].copy()
            gt_sub = gt_sub.sort_values(key).reset_index(drop=True)
            pred_sub = pred_sub.sort_values(key).reset_index(drop=True)
        else:
            # 模糊匹配 group by列默认是1对1匹配。
            advanced_matcher = create_advanced_group_by_matcher()
            gt_idx_used = set()
            pred_idx_used = set()
            gt_indices = gt.index.tolist()
            pred_indices = pred.index.tolist()
            gt_matched = []
            pred_matched = []
            for gt_idx in gt_indices:
                gt_val = gt.at[gt_idx, key]
                for pred_idx in pred_indices:
                    if pred_idx in pred_idx_used:
                        continue
                    pred_val = pred.at[pred_idx, key]
                    is_match, _ = advanced_matcher.match_values_by_type(gt_val, pred_val)
                    if is_match:
                        gt_matched.append(gt_idx)
                        pred_matched.append(pred_idx)
                        pred_idx_used.add(pred_idx)
                        break  # 1对1配对
            gt_sub = gt.loc[gt_matched].copy().reset_index(drop=True)
            pred_sub = pred.loc[pred_matched].copy().reset_index(drop=True)
    else:
        # 多主键，精确匹配
        gt['__merge_key__'] = gt[primary_keys].astype(str).agg('_'.join, axis=1)
        pred['__merge_key__'] = pred[primary_keys].astype(str).agg('_'.join, axis=1)
        keys_inter = set(gt['__merge_key__']) & set(pred['__merge_key__'])
        gt_sub = gt[gt['__merge_key__'].isin(keys_inter)].copy()
        pred_sub = pred[pred['__merge_key__'].isin(keys_inter)].copy()
        gt_sub = gt_sub.sort_values(primary_keys).reset_index(drop=True)
        pred_sub = pred_sub.sort_values(primary_keys).reset_index(drop=True)
    return gt_sub, pred_sub


class GroundTruthSQLExecutor:
    """
    用pandas DataFrame模拟SQL的基本操作（投影、过滤、聚合、连接），
    适配属性抽取系统的测试需求。
    """

    def __init__(self, tables: Dict[str, pd.DataFrame], attr_types: Dict[str, List[str]] = None, primary_keys: List[str] = ["id"]):
        """
        :param tables: {'table_name': pd.DataFrame}
        :param attr_types: 字段属性类型，如 {"disease": ["multi", "fixed"], "age": ["single", "fixed"]}
        """
        self.primary_keys = primary_keys
        self.tables = {k.lower(): v for k, v in tables.items()}
        self.attr_types = attr_types if attr_types is not None else {} # 默认为空，因为当前所有的字符串类型都默认为多值属性。

    def parse_sql(self, sql: str):
        """
        粗略解析SQL为 select, from, where, group by, join, aggregation等信息
        适配简单SQL。复杂SQL需自行扩展。
        """
        sql = sql.strip().rstrip(';')
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.I)
        from_match = re.search(r"FROM\s+([a-zA-Z0-9_]+)", sql, re.I)
        where_match = re.search(r"WHERE\s+(.*?)(GROUP\s+BY|ORDER\s+BY|$)", sql, re.I)
        group_by_match = re.search(r"GROUP\s+BY\s+(.*?)(ORDER\s+BY|$)", sql, re.I)
        # join_match = re.search(r"JOIN\s+([a-zA-Z0-9_]+)\s+ON\s+(.+?)(WHERE|GROUP\s+BY|ORDER\s+BY|$)", sql, re.I)
        join_match = re.search(
            r"JOIN\s+([a-zA-Z0-9_]+)\s+ON\s+(.+?)(?=\s+WHERE|\s+GROUP\s+BY|\s+ORDER\s+BY|;|$)",
            sql, re.I | re.S
        )
        agg_match = re.findall(r"(MAX|MIN|AVG|SUM|COUNT)\s*\((.*?)\)", sql, re.I)

        return {
            'select': select_match.group(1).lower().strip() if select_match else '*',
            'from': from_match.group(1) if from_match else None,
            'where': where_match.group(1).strip() if where_match else None,
            'group_by': group_by_match.group(1).strip() if group_by_match else None,
            'join': {
                'table': join_match.group(1),
                'on': join_match.group(2).strip()
            } if join_match else None,
            'agg': agg_match if agg_match else None
        }

    def _handle_where(self, df: pd.DataFrame, where_clause: str) -> pd.Series:
        """
        pandas实现自定义SQL where子句，字符串==实现为IN
        支持多值型、单值型的判定。
        """
        if not where_clause:
            return pd.Series([True] * len(df))
        df = df.rename(columns={col: col.split('.', 1)[-1] for col in df.columns})
        
        # 支持 AND/OR，多条件处理
        # 注意: AND/OR 不区分大小写
        where_clause = where_clause.replace('AND', '&').replace('and', '&')
        where_clause = re.sub(r'\bOR\b', '|', where_clause, flags=re.IGNORECASE)
        where_clause = re.sub(r'\b[a-zA-Z0-9_]+\.', '', where_clause)

        # 定义pattern，支持数值/日期/字符串
        pattern = re.compile(
            r"(\w+)\s*(==|=|!=|<>)\s*'([^']*)'"
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*(\d{4}/\d{1,2}/\d{1,2})"
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)"
        )

        def cond_replacer(match):
            # 与get_gt参考实现类似
            if match.group(1):  # 字符串
                col, op, val = match.group(1).lower(), match.group(2), match.group(3)
                attr_type = self.attr_types.get(col, ['multi', 'fixed'])
                if attr_type[0] == 'single':
                    # 字符串比较：精确等值
                    if op in ['==', '=']:
                        return f"(df['{col}'].fillna('').str.strip() == '{val}')"
                    elif op in ['!=', '<>']:
                        return f"(df['{col}'].fillna('').str.strip() != '{val}')"
                    else:
                        return f"(df['{col}'].fillna('').str.strip() {op} '{val}')"
                else:
                    # 多值型，in
                    if op in ['==', '=']:
                        return f"(df['{col}'].fillna('').apply(lambda x: '{val}' in str(x)))"
                    elif op in ['!=', '<>']:
                        return f"(df['{col}'].fillna('').apply(lambda x: '{val}' not in str(x)))"
                    else:
                        return f"(df['{col}'].fillna('').apply(lambda x: str(x) {op} '{val}'))"
            elif match.group(4):  # 日期
                col, op, val = match.group(4).lower(), match.group(5), match.group(6)
                return f"(pd.to_datetime(df['{col}'], errors='coerce', format='mixed') {op} pd.to_datetime('{val}', errors='coerce', format='mixed'))"
            elif match.group(7):  # 数字
                col, op, val = match.group(7).lower(), match.group(8), match.group(9)
                return f"(pd.to_numeric(df['{col}'], errors='coerce').fillna(0) {op} {val})"
            else:
                raise ValueError("未知SQL条件匹配")
        
        where_py = pattern.sub(cond_replacer, where_clause)
        # 兼容裸日期格式
        where_py = re.sub(r"(?<!['\w])(\d{4}/\d{1,2}/\d{1,2})(?!['\w])", r"'\1'", where_py)
        # print(f"Python where: {where_py}")
        cond = eval(where_py)
        return cond

    def _deprecated_handle_select(self, df: pd.DataFrame, select_cols: str) -> pd.DataFrame:
        """
        投影操作。支持 select * 或逗号分隔字段
        """
        select_cols = [c.strip().lower() for c in select_cols.split(',')] if select_cols != '*' else list(df.columns)
        return df[select_cols]

    def _handle_select(self, df: pd.DataFrame, select_cols: str) -> pd.DataFrame:
        """
        投影操作。支持 select * 或逗号分隔字段，且始终保留主键列
        """
        if select_cols != '*':
            select_cols = [c.strip().lower() for c in select_cols.split(',')]
        else:
            select_cols = list(df.columns)
        # 保证主键列在select_cols中
        for pk in self.primary_keys:
            if pk not in select_cols and pk in df.columns:
                select_cols.insert(0, pk)  # 可插入到最前，也可append到最后
        # 去重且保持顺序
        seen = set()
        select_cols = [x for x in select_cols if not (x in seen or seen.add(x))]
        return df[select_cols]

    def _handle_groupby_agg(self, df: pd.DataFrame, group_by: Optional[str], agg: Optional[List[Tuple[str, str]]]) -> pd.DataFrame:
        """
        group by + agg
        agg: [(聚合函数, 字段名)], 如 [("MAX", "age")]
        """
        if not group_by or not agg:
            return df
        
        # 精细清洗：按空白符split再用单空格join，防止多余空格
        if group_col in df.columns and df[group_col].dtype == object:
            df[group_col] = df[group_col].astype(str).apply(lambda x: " ".join(x.split()))

        
        group_col = group_by.strip().lower()
        agg_ops = {}
        for func, col in agg:
            func = func.upper()
            col = col.strip().lower()
            if func == "COUNT" and col == "*":
                # COUNT(*) 特殊处理
                agg_col_name = "count(*)"
                df[agg_col_name] = 1
                agg_ops[agg_col_name] = ("count", agg_col_name)
            elif func == "COUNT":
                agg_col_name = f"count({col})"
                # 废弃：把df的列col复制到列agg_col_name上
                # df[agg_col_name] = df[col]
                agg_ops[agg_col_name] = ("count", col)
            else:
                func_map = {
                    "MAX": "max",
                    "MIN": "min",
                    "AVG": "mean",
                    "SUM": "sum"
                }
                agg_func = func_map[func]
                mod_agg_func_name = agg_func
                if agg_func == "mean":
                    mod_agg_func_name = "avg"
                agg_col_name = f"{mod_agg_func_name}({col})"

                agg_ops[agg_col_name] = (agg_func, col)
        
        # 构造agg字典
        agg_dict = {new_col: (col, func) for new_col, (func, col) in agg_ops.items()}
        # groupby聚合
        res = df.groupby(group_col).agg(**agg_dict).reset_index()
        return res


    def _deprecated_handle_groupby_agg(self, df: pd.DataFrame, group_by: Optional[str], agg: Optional[List[Tuple[str, str]]]) -> pd.DataFrame:
        """
        group by + agg
        agg: [(聚合函数, 字段名)], 如 [("MAX", "age")]
        """
        if not group_by or not agg:
            return df
        
        group_col = group_by.strip().lower()
        # 只支持单字段groupby，可扩展
        agg_func, agg_col = agg[0][0].upper(), agg[0][1].strip().lower()

        # 空/非法值清洗：略，可按需要扩展
        
        if agg_func == "COUNT" and agg_col == "*":
            res = df.groupby(group_col).size().reset_index(name="COUNT")
        elif agg_func == "COUNT":
            res = df.groupby(group_col)[agg_col].count().reset_index(name="COUNT")
        else:
            agg_map = {
                "MAX": "max",
                "MIN": "min",
                "AVG": "mean",
                "SUM": "sum"
            }
            res = df.groupby(group_col)[agg_col].agg(agg_map[agg_func]).reset_index()
        return res

    def _v0_handle_join(self, df_left: pd.DataFrame, df_right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
        """
        字符串类型的 == 实现为in（模糊join），支持多表join
        on格式: "left_col == right_col"
        """
        # 只支持一对连接列，可自行扩展
        m = re.match(r"(\w+)\s*(==|=)\s*(\w+)", on.strip())
        if not m:
            raise ValueError(f"JOIN ON语句格式不合法: {on}")
        left_col, op, right_col = m.group(1).lower(), m.group(2), m.group(3).lower()

        # 如果是字符串列，采用in模糊join
        # 此处简单实现，不依赖外部table matcher
        def fuzzy_merge(left, right, left_on, right_on):
            # 左表每一行遍历右表找in
            res = []
            for idx, lrow in left.iterrows():
                lval = str(lrow[left_on]) if pd.notnull(lrow[left_on]) else ""
                match_idx = right[right[right_on].astype(str).apply(lambda x: lval in x if pd.notnull(x) else False)].index
                if len(match_idx) > 0:
                    # 只保留第一个匹配（可多对多/全展开自行扩展）
                    for rid in match_idx:
                        merged_row = pd.concat([lrow, right.loc[rid]], axis=0)
                        res.append(merged_row)
            if not res:
                return pd.DataFrame(columns=list(left.columns)+list(right.columns))
            merged = pd.DataFrame(res)
            # 去重列
            merged = merged.loc[:, ~merged.columns.duplicated()]
            return merged.reset_index(drop=True)
        
        # 判断类型
        if pd.api.types.is_string_dtype(df_left[left_col]) or pd.api.types.is_string_dtype(df_right[right_col]):
            joined = fuzzy_merge(df_left, df_right, left_col, right_col)
        else:
            joined = pd.merge(df_left, df_right, left_on=left_col, right_on=right_col, how=how)        
        # 新增：为所有列加表名前缀
        left_table_name = [k for k, v in self.tables.items() if v.equals(df_left)][0]
        right_table_name = [k for k, v in self.tables.items() if v.equals(df_right)][0]
        joined = joined.rename(columns={col: f"{left_table_name}.{col}" for col in df_left.columns if col in joined.columns})
        joined = joined.rename(columns={col: f"{right_table_name}.{col}" for col in df_right.columns if col in joined.columns})
        return joined        


    def _handle_join(self, df_left, df_right, on, left_table_name, right_table_name, how="inner"):

        def fuzzy_merge(left, right, left_on, right_on):
            # 左表每一行遍历右表找in
            res = []
            for idx, lrow in left.iterrows():
                lval = str(lrow[left_on]) if pd.notnull(lrow[left_on]) else ""
                match_idx = right[right[right_on].astype(str).apply(lambda x: lval in x if pd.notnull(x) else False)].index
                if len(match_idx) > 0:
                    # 只保留第一个匹配（可多对多/全展开自行扩展）
                    for rid in match_idx:
                        merged_row = pd.concat([lrow, right.loc[rid]], axis=0)
                        res.append(merged_row)
            if not res:
                return pd.DataFrame(columns=list(left.columns)+list(right.columns))
            merged = pd.DataFrame(res)
            # 去重列
            merged = merged.loc[:, ~merged.columns.duplicated()]
            return merged.reset_index(drop=True)
        

        # 给所有列加前缀
        df_left = df_left.rename(columns={col: f"{left_table_name}.{col}" for col in df_left.columns})
        df_right = df_right.rename(columns={col: f"{right_table_name}.{col}" for col in df_right.columns})

        # primary_keys的列要设置为"{table_name}.id"的形式
        self.primary_keys = [f"{left_table_name}.id", f"{right_table_name}.id"]

        on = on.lower()
        # 解析 on 条件，支持 table.col = table.col
        m = re.match(r"(?:(\w+)\.)?(\w+)\s*(==|=)\s*(?:(\w+)\.)?(\w+)", on.strip())
        if not m:
            raise ValueError(f"JOIN ON语句格式不合法: {on}")
        left_tbl, left_col, op, right_tbl, right_col = m.groups()
        left_col_full = f"{left_tbl}.{left_col}"
        right_col_full = f"{right_tbl}.{right_col}"

        if left_table_name != left_tbl:
            df_tem = df_left
            df_left = df_right
            df_right = df_tem
            
        # jztodebug join的左表和右表名要根据on语句中的位置对应到df_left和df_right上？
        # right_table_name = right_tbl

        # 判断类型
        if pd.api.types.is_string_dtype(df_left[left_col_full]) or pd.api.types.is_string_dtype(df_right[right_col_full]):
            joined = fuzzy_merge(df_left, df_right, left_col_full, right_col_full)
        else:
            joined = pd.merge(df_left, df_right, left_on=left_col_full, right_on=right_col_full, how=how)
        return joined

    def _split_where_by_table(self, where_clause, main_table, join_table):
        """
        拆分where条件，返回三部分：
        - 只涉及主表的条件
        - 只涉及JOIN表的条件
        - 需要JOIN后才能处理的条件
        """
        # 简单实现：按AND拆分，再用正则判断字段前缀
        if not where_clause:
            return None, None, None
        conds = re.split(r'\s+AND\s+|\s+and\s+', where_clause)
        main_conds, join_conds, post_join_conds = [], [], []
        for cond in conds:
            # 用更严格的正则，完整匹配字段名（包括下划线）
            fields = re.findall(r'([a-zA-Z_][\w]*\.[a-zA-Z_][\w]*)', cond)
            if not fields:
                # 没有表前缀，默认主表
                main_conds.append(cond)
            else:
                tables = set(f.split('.')[0].lower() for f in fields)
                if len(tables) == 1:
                    if list(tables)[0] == main_table:
                        main_conds.append(cond)
                    elif list(tables)[0] == join_table:
                        join_conds.append(cond)
                    else:
                        post_join_conds.append(cond)
                else:
                    post_join_conds.append(cond)
        return ' AND '.join(main_conds), ' AND '.join(join_conds), ' AND '.join(post_join_conds)

    def v0__split_where_by_table(self, where_clause, main_table, join_table):
        """
        拆分where条件，返回三部分：
        - 只涉及主表的条件
        - 只涉及JOIN表的条件
        - 需要JOIN后才能处理的条件
        """
        # 简单实现：按AND拆分，再用正则判断字段前缀
        if not where_clause:
            return None, None, None
        conds = re.split(r'\s+AND\s+|\s+and\s+', where_clause)
        main_conds, join_conds, post_join_conds = [], [], []
        for cond in conds:
            # 判断字段属于哪个表
            fields = re.findall(r'(\w+\.\w+)', cond)
            if not fields:
                # 没有表前缀，默认主表
                main_conds.append(cond)
            else:
                tables = set(f.split('.')[0].lower() for f in fields)
                if len(tables) == 1:
                    if list(tables)[0] == main_table:
                        main_conds.append(cond)
                    elif list(tables)[0] == join_table:
                        join_conds.append(cond)
                    else:
                        post_join_conds.append(cond)
                else:
                    post_join_conds.append(cond)
        return ' AND '.join(main_conds), ' AND '.join(join_conds), ' AND '.join(post_join_conds)


    def execute(self, sql: str) -> pd.DataFrame:
        parsed = self.parse_sql(sql)
        main_table_name = parsed['from'].lower()
        main_table = self.tables.get(main_table_name)
        df = main_table.copy()

        join_table_name = parsed['join']['table'].lower() if parsed['join'] else None
        join_table = self.tables.get(join_table_name) if join_table_name else None

        # 新增：where条件拆分
        main_where, join_where, post_join_where = self._split_where_by_table(
            parsed['where'], main_table_name, join_table_name
        )

        # 先对主表做where
        if main_where:
            cond = self._handle_where(df, main_where)
            df = df.loc[cond]

        # join表也做where
        if join_table is not None and join_where:
            cond = self._handle_where(join_table, join_where)
            join_table = join_table.loc[cond]

        # join
        if parsed['join']:
            # self.primary_keys =  这里会在handle_join里面进行修改。
            df = self._handle_join(df, join_table, parsed['join']['on'], main_table_name, join_table_name)

        # join后再做剩余where
        # if post_join_where:
        #     cond = self._handle_where(df, post_join_where)
        #     df = df.loc[cond]

        # group by + agg
        if parsed['group_by'] and parsed['agg']:
            self.primary_keys = [parsed['group_by']]
            df = self._handle_groupby_agg(df, parsed['group_by'], parsed['agg'])

        # select
        df = self._handle_select(df, parsed['select'])
        return df.reset_index(drop=True)
    # ...existing code...


import pandas as pd
from typing import List, Tuple, Dict

class AccEvaluator:
    @staticmethod
    def safe_split(cell, sep="||"):
        """安全切分单元格（多值属性支持）"""
        if pd.isna(cell) or str(cell).strip() == "":
            return []
        return [s.strip() for s in str(cell).split(sep)]


    @staticmethod
    def _llm_semantic_match(
        pred_lists: List[List[str]],
        gt_lists: List[List[str]],
        batch_size: int = 1,
        model: str = "gpt-4.1-nano",
    ) -> List[int]:
        """
        LLM批量判等计数：每组(a_list, b_list)计算a_list中有多少元素能在b_list中找到语义等价项。
        参数：
            a_lists, b_lists: 均为二维list，长度需一致，每一对做一次判等
            batch_size: LLM请求批次大小
            model: LLM模型名
        返回：
            List[int]: 每组的匹配个数（如[2,0,1...]）
        """
        from litellm import batch_completion

        assert len(pred_lists) == len(gt_lists), "a_lists和b_lists长度必须一致"
        prompts = []
        for pred_list, gt_list in zip(pred_lists, gt_lists):
            prompt = [
                {
                    "role": "system",
                    "content": "You are an expert in evaluating semantic similarity between terms. Respond strictly as instructed. Do not explain, infer, or expand. Begin your response immediately."
                },
                {
                    "role": "user",
                    "content": f"""Given two lists of terms, compare each term in List A with all terms in List B. For each term in List A, determine if there is any term in List B that expresses the same meaning (even if the words are different).
- Count how many terms in List A have at least one semantically similar match in List B.
- if number in List, If there are numerical values, a small margin of error within 1% is allowed.
- Respond with ONLY the final count (an integer), with nothing else.
- Do not explain, analyze, or add any reasoning.
- Begin your response immediately with the integer.

List A: {pred_list}
List B: {gt_list}
"""
                }
            ]
            prompts.append(prompt)

        results = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            responses = batch_completion(
                model=model,
                messages=prompts[i:i+batch_size],
                stop=None,
                max_tokens=32,
                temperature=0,
            )
            for response in responses:
                content = response['choices'][0]['message']['content'].strip()
                # 防止llm输出非数字
                try:
                    count = int(content)
                except Exception:
                    count = 0
                results.append(count)
        # 以f1作为作为最终的count值，如果是单值属性，f1值就只有0或1，如果是多值属性，f1值在0到1之间
        # 按顺序算出f1值作为最终的results
        for i in range(len(results)):
            p = results[i] / len(pred_lists[i]) if len(pred_lists[i]) > 0 else 0
            r = results[i] / len(gt_lists[i]) if len(gt_lists[i]) > 0 else 0
            if p + r == 0:
                results[i] = 0
            else:
                results[i] = 2 * p * r / (p + r)

        return results



    @staticmethod
    def _mock_semantic_match(pred_list: List[str], gt_list: List[str]) -> int:
        """
        mock 版，直接用 set 交集，实际部署时替换为 LLM 调用或更复杂逻辑。
        返回a_list中有多少个元素在b_list中有"等价表达"。
        """
        # 这其实是一个多值单元格的匹配，多值用集合的形式表示，以f1值作为最终的准确率指标
        intersect_len = len(set([i.lower() for i in pred_list]) & set([j.lower() for j in gt_list]))
        p = intersect_len / len(pred_list) if len(pred_list) > 0 else 0
        r = intersect_len / len(gt_list) if len(gt_list) > 0 else 0
        
        f1_val =  2 * p * r / (p + r) if (p + r) > 0 else 0
        return f1_val

    @staticmethod
    def compute_col_prf(gt_col: pd.Series, pred_col: pd.Series, gt_len: int, pred_len: int) -> Tuple[float, float, float]:
        """
        按列计算 precision/recall/f1（考虑多值字段，LLM批量判等）。
        """
        # 全部组装成二维list
        gt_lists = [AccEvaluator.safe_split(val) for val in gt_col]
        pred_lists = [AccEvaluator.safe_split(val) for val in pred_col]
        # 用llm批量判等
        match_f1_list = AccEvaluator._llm_semantic_match(pred_lists, gt_lists)
        matched_count = sum(match_f1_list)
        gt_count = gt_len
        pred_count = pred_len

        # 容错: 若一列全是空值，分母可为0
        precision = matched_count / pred_count if pred_count > 0 else 0.0
        recall = matched_count / gt_count if gt_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1


    @staticmethod
    def compute_acc(primary_keys: List[str], gt: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, List]:
        """
        按主键intersection + 每列统计，计算全表pr, r, f1。
        返回字典格式: {'p': [...], 'r': [...], 'f1': [...], 'col': [...]}
        """
        result = {"p": [], "r": [], "f1": [], "col": []}

        # 清理主键列，确保str类型
        for key in primary_keys:
            if key in gt.columns:
                gt[key] = gt[key].astype(str)
            if key in pred.columns:
                pred[key] = pred[key].astype(str)

        # # 求主键intersection
        gt_sub, pred_sub = primary_key_intersect(primary_keys, gt, pred)
        # if len(primary_keys) == 1:
        #     key = primary_keys[0]
        #     keys_inter = set(gt[key]) & set(pred[key])
        #     gt_sub = gt[gt[key].isin(keys_inter)].copy()
        #     pred_sub = pred[pred[key].isin(keys_inter)].copy()
        # else:
        #     # 多主键
        #     gt['__merge_key__'] = gt[primary_keys].astype(str).agg('_'.join, axis=1)
        #     pred['__merge_key__'] = pred[primary_keys].astype(str).agg('_'.join, axis=1)
        #     keys_inter = set(gt['__merge_key__']) & set(pred['__merge_key__'])
        #     gt_sub = gt[gt['__merge_key__'].isin(keys_inter)].copy()
        #     pred_sub = pred[pred['__merge_key__'].isin(keys_inter)].copy()
        #     # 用于排序
        #     gt_sub = gt_sub.sort_values(primary_keys).reset_index(drop=True)
        #     pred_sub = pred_sub.sort_values(primary_keys).reset_index(drop=True)

        # # 按主键排序对齐
        # if len(primary_keys) == 1:
        #     key = primary_keys[0]
        #     gt_sub = gt_sub.sort_values(key).reset_index(drop=True)
        #     pred_sub = pred_sub.sort_values(key).reset_index(drop=True)

        # 所有要比对的列（不含主键）
        columns_to_eval = [col for col in gt_sub.columns if col not in primary_keys and col != '__merge_key__']


        # 去除重复主键
        gt_sub = drop_duplicate_primary_keys_rows(gt_sub, primary_keys)
        pred_sub = drop_duplicate_primary_keys_rows(pred_sub, primary_keys)

        # 保证 pred_sub, gt_sub 行数一致
        # assert len(gt_sub) == len(pred_sub), "intersection部分gt和pred行数不一致，主键未对齐！"

        gt_len = len(gt)
        pred_len = len(pred)
        for col in columns_to_eval:
            # 一一对应，计算prf
            p, r, f1 = AccEvaluator.compute_col_prf(gt_sub[col], pred_sub[col], gt_len, pred_len)
            result['p'].append(p)
            result['r'].append(r)
            result['f1'].append(f1)
            result['col'].append(col)
        return result


def clean_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗extract表格，统一主键命名与格式，处理空值和类型异常，并将所有列名转为小写。
    1. 所有列名转为小写；
    2. file_name 列改为 id，并去掉后缀(.txt/.pdf/.md等)；
    3. id 列去小数点（如1.0->1），并全部转为字符串；
    4. 全部空值填充为空串。
    """
    df = df.copy()
    # 1. 所有列名转小写
    df.columns = df.columns.str.lower()
    # 2. file_name 列处理
    if 'file_name' in df.columns:
        # 下面是pred_table为单表的情况
        df = df.rename(columns={'file_name': 'id'})
        df['id'] = df['id'].astype(str).str.replace(r'\.(txt|pdf|md)$', '', regex=True)

    # 需要加入pred_table为多表的情况，比如team.file_name
    # 先用正则表达式匹配形如'table_name.file_name'这种列名，再重命名成'table_name.id'
    df = df.rename(columns={col: col.replace('file_name', 'id') for col in df.columns if 'file_name' in col})

        
    # 3. 处理所有以 .id 结尾的列（包括 id 本身）
    for col in df.columns:
        if col.endswith('.id') or col == 'id':
            # 去掉.txt/.pdf/.md等后缀
            df[col] = df[col].astype(str).str.replace(r'\.(txt|pdf|md)$', '', regex=True)
            # 去掉小数点（如1.0->1）
            df[col] = df[col].str.replace(r'\.0+$', '', regex=True)
            # 空值填充为空串
            df[col] = df[col].replace('nan', '').replace('None', '')
            # 不强制转int，直接按字符串排序
            df = df.sort_values(by=col, ignore_index=True)
            df[col] = df[col].astype(str)    
        
    # 3. id列去小数点转字符串:如果是浮点数的话；如果不是浮点数，就无所谓?? jztonote id列到底允许怎样的格式？
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str).str.replace(r'\.0+$', '', regex=True)
    # 4. 空值填充为空串
    df = df.fillna("")
    # 5. id列转成整数，按从小到大排序，之后再转回字符串 jznote 对于非整数类型的file_name id，不能转成整数，应该直接按照字符串排序。
    if 'id' in df.columns:
        df['id'] = df['id'].astype(int)
        df = df.sort_values(by='id', ignore_index=True)
        df['id'] = df['id'].astype(str)
        # 对于aggregation的表,id本身就可能不存在

    
    return df


def cal_sql_acc(sql: str, gt_tables_dict: Dict[str, pd.DataFrame], pred_table: pd.DataFrame):
    # 送进来的table可能没有clean, 需要遍历gt_tables_dict对表格进行clean
    for table_name, table in gt_tables_dict.items():
        gt_tables_dict[table_name] = clean_pd(table)
    pred_table = clean_pd(pred_table)
    # join时，gt_table_dict有2个表，否则就一个。
    sql_executor = GroundTruthSQLExecutor( gt_tables_dict )
    filtered_gt_table = sql_executor.execute(sql)
    filtered_gt_table_copy = filtered_gt_table.copy()
    primary_keys = sql_executor.primary_keys
    acc_dict = AccEvaluator.compute_acc(primary_keys, filtered_gt_table, pred_table)
    # 利用acc_dict算出整张表的准确率，取列名为avg_acc，加入字典中
    avg_acc_dict = {}
    for col in ['p', 'r', 'f1']:
        avg_acc_dict['avg_{}'.format(col)] = np.mean(acc_dict[col])
    acc_dict['col'].append('avg_acc')
    acc_dict['p'].append(avg_acc_dict['avg_p'])
    acc_dict['r'].append(avg_acc_dict['avg_r'])
    acc_dict['f1'].append(avg_acc_dict['avg_f1'])
    return acc_dict, filtered_gt_table_copy



