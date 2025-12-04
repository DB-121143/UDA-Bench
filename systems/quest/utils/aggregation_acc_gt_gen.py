import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

import tiktoken

GT_file_path = "/data/QUEST/jzshe/project/quest/data/benchmark/ground_truth/player.csv"
SAVE_file_dir = "/data/QUEST/jzshe/project/quest/quest/tests/log/player/agg_gt/"
os.makedirs(SAVE_file_dir, exist_ok=True)

# 对应文件名
raw_file_names= [
    "query1.csv", "query2.csv", "query3.csv", "query4.csv", "query5.csv",
    "query6.csv", "query7.csv", "query8.csv", "query9.csv", "query10.csv"
]

save_filenames = []

for  raw_file_name in raw_file_names:
    save_filenames.append(SAVE_file_dir + raw_file_name)

# 载入数据
df = pd.read_csv(GT_file_path)

# 新增：值清洗函数
def clean_group_value(value):
    """
    清洗group by列的值：
    1. 去掉首尾空白符
    2. 正则表达式匹配去掉token间多余空白符
    """
    if pd.isnull(value) or value is None:
        return value
    
    if not isinstance(value, str):
        value = str(value)
    
    # 去掉首尾空白符
    value = value.strip()
    
    # 如果清洗后为空字符串，返回None
    if value == "":
        return None
    
    import re
    cleaned_value = re.sub(r'\s+', ' ', value).strip()
    return cleaned_value if cleaned_value else None


# 空/非法值判定函数
def is_invalid(value, col, dtype):
    if pd.isnull(value):
        return True
    if isinstance(value, str):
        if value.strip() == "" or value.strip().lower() in ["unknown", "n/a", "--"]:
            return True
    if col == "draft_pick":
        # draft_pick为0算空
        try:
            if int(value) == 0:
                return True
        except:
            return True
    if dtype == "int" or dtype == "float":
        try:
            float(value)
        except:
            return True
    if dtype == "date":
        try:
            if pd.isnull(pd.to_datetime(value, errors='coerce')):
                return True
        except:
            return True
    return False

# 字段类型映射（手工维护）
column_types = {
    "draft_year": "int",
    "college": "str",
    "mvp_awards": "int",
    "birth_date": "date",
    "fiba_world_cup": "int",
    "draft_pick": "int",
    "position": "str",
    "nationality": "str",
    "team": "str",
    "nba_championships": "int",
    "age": "int",
    "name": "str",
    "olympic_gold_medals": "int",
}

# SQL语句列表
queries = [
    ("college", "MAX", "draft_year"),
    ("position", "COUNT", "draft_pick"),
    ("position", "MIN", "nba_championships"),
    ("college", "AVG", "age"),
    ("team", "COUNT", "*"),
    ("college", "SUM", "age"),
    ("position", "AVG", "age"),
    ("college", "AVG", "draft_year"),
    ("position", "MAX", "age"),
    ("position", "COUNT", "*"),
]



# 针对每个查询单独处理
for i, (group_col, agg_func, agg_col) in enumerate(queries):
    # 复制原数据
    df_query = df.copy()

    # 标准化非法/空值
    # group_col
    group_dtype = column_types.get(group_col, "str")
    # df_query[group_col] = df_query[group_col].apply(lambda x: None if is_invalid(x, group_col, group_dtype) else x)
    df_query[group_col] = df_query[group_col].apply(lambda x: clean_group_value(x) if not is_invalid(x, group_col, group_dtype) else None)
    # agg_col（如果不是*和COUNT）
    if agg_func != "COUNT" or agg_col != "*":
        agg_dtype = column_types.get(agg_col, "str")
        df_query[agg_col] = df_query[agg_col].apply(lambda x: None if is_invalid(x, agg_col, agg_dtype) else x)

    # 特殊：如果是COUNT(*)，聚合字段随便设
    if agg_func == "COUNT" and agg_col == "*":
        agg_res = df_query.groupby(group_col).size().reset_index(name="COUNT")
        agg_res[group_col] = agg_res[group_col].where(agg_res[group_col].notnull(), None)
        agg_res["COUNT"] = agg_res["COUNT"].where(agg_res[group_col].notnull(), None)
    elif agg_func == "COUNT":
        # COUNT(draft_pick) 非法为None
        def count_valid(x):
            # 只统计非空且非非法的draft_pick
            return x.notnull().sum() if x.name == group_col else None
        agg_res = df_query.groupby(group_col)[agg_col].count().reset_index(name="COUNT")
        # None组设为None
        agg_res["COUNT"] = agg_res.apply(lambda row: None if row[group_col] is None else row["COUNT"], axis=1)
        agg_res[group_col] = agg_res[group_col].where(agg_res[group_col].notnull(), None)
    else:
        # 其余聚合
        agg_op = {
            "MAX": lambda x: None if x.name == group_col or x.isnull().all() else x.max(),
            "MIN": lambda x: None if x.name == group_col or x.isnull().all() else x.min(),
            "AVG": lambda x: None if x.name == group_col or x.isnull().all() else x.mean(),
            "SUM": lambda x: None if x.name == group_col or x.isnull().all() else x.sum()
        }[agg_func]
        agg_res = df_query.groupby(group_col)[agg_col].agg(agg_op).reset_index()
        # None组聚合结果设为None
        agg_res[agg_col] = agg_res.apply(lambda row: None if row[group_col] is None else row[agg_col], axis=1)
        agg_res[group_col] = agg_res[group_col].where(agg_res[group_col].notnull(), None)

    # 导出csv
    agg_res.to_csv(save_filenames[i], index=False)

print("All queries finished, csv files generated.")
