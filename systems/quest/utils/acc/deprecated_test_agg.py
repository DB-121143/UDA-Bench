#%%
import pandas as pd
import re
import os
from tqdm import tqdm
import json


def get_gt(sql, attr_types, df, SQL):
    # 解析SELECT列
    select_col = re.search(r'SELECT (.+?)\s+FROM', sql, re.I).group(1).strip()
    select_cols = [c.strip().lower() for c in select_col.split(',')]
    agg_funcs = {}

    for col in select_cols:
        match = re.match(r'(count|min|max|avg|sum)\((.*?)\)', col.strip(), re.I)
        if match:
            func = match.group(1).lower()     
            arg = match.group(2).strip().lower() 
            agg_funcs[arg] = func
    group_col = re.search(r'GROUP BY\s+([A-Za-z_][A-Za-z0-9_]*);?', sql, re.I).group(1).lower()


    def get_result(df, agg, group, agg_col):
        if agg_col != '*':
            if pd.api.types.is_numeric_dtype(df[agg_col]):
                df.loc[:, agg_col]= pd.to_numeric(df[agg_col], errors='coerce')
        if agg.lower() == 'min':
            df = df.dropna(subset=[agg_col])
            res = df.groupby(group)[agg_col].min().reset_index()
            res.rename(columns={agg_col: "min"}, inplace=True)
            return res
        elif agg.lower() == 'max':
            df = df.dropna(subset=[agg_col])
            res = df.groupby(group)[agg_col].max().reset_index()
            res.rename(columns={agg_col: "max"}, inplace=True)
            return res
        elif agg.lower() == 'sum':
            df = df.dropna(subset=[agg_col])
            res = df.groupby(group)[agg_col].sum().reset_index()
            res.rename(columns={agg_col: "sum"}, inplace=True)
            return res
        elif agg.lower() == 'avg':
            df = df.dropna(subset=[agg_col])
            res = df.groupby(group)[agg_col].mean().reset_index()
            res.rename(columns={agg_col: "avg"}, inplace=True)
            return res
        else:
            if agg_col != "*":
                df = df.dropna(subset=[agg_col])
                res = df.groupby(group)[agg_col].count().reset_index()
                res.rename(columns={agg_col: "count"}, inplace=True)
                return res
            else:
                res = df.groupby(group).size().reset_index(name="count")
                return res
            

    def sql_condition_replace(match):
        # 判断类型顺序：数字 -> 日期 -> 字符串
        if match.group(1):  # 带引号字符串
            col = match.group(1)
            op = match.group(2)
            val = match.group(3)
            col_lower = col.lower()
            attr_type = attr_types.get(col_lower, ['single', 'fixed'])
            is_numeric = False
            is_date = False
            val_out = f"'{val}'"
            if not pd.api.types.is_string_dtype(df[col_lower]):
                df[col_lower] = df[col_lower].astype(str)
        elif match.group(4):  # 裸日期
            col = match.group(4)
            op = match.group(5)
            val = match.group(6)
            col_lower = col.lower()
            attr_type = attr_types.get(col_lower, ['single', 'fixed'])
            is_numeric = False
            is_date = True
            val_out = f"'{val}'"
            if not pd.api.types.is_datetime64_any_dtype(df[col_lower]):
                df[col_lower] = pd.to_datetime(df[col_lower], errors='coerce', format='mixed')
        elif match.group(7):  # 纯数字
            col = match.group(7)
            op = match.group(8)
            val = match.group(9)
            col_lower = col.lower()
            attr_type = attr_types.get(col_lower, ['single', 'fixed'])
            is_numeric = True
            is_date = False
            val_out = f"{val}"
            if not pd.api.types.is_numeric_dtype(df[col_lower]):
                df[col_lower] = pd.to_numeric(df[col_lower], errors='coerce')
            df[col_lower] = df[col_lower].fillna(0)
            
        else:
            raise ValueError(f"未知的SQL条件匹配：{match.groups()}")

        if attr_type[0] == 'single':
            if is_numeric:
                # 数值型字段
                return f"(df['{col_lower}'] {op} {val_out})"
            elif is_date:
                # 日期型字段
                return f"(pd.to_datetime(df['{col_lower}'], errors='coerce', format='mixed') {op} pd.to_datetime({val_out}, errors='coerce', format='mixed'))"
            else:
                # 字符串型字段
                if op in ['==', '=']:
                    return f"(df['{col_lower}'].fillna('').str.strip() == {val_out})"
                elif op in ['!=', '<>']:
                    return f"(df['{col_lower}'].fillna('').str.strip() != {val_out})"
                else:
                    return f"(df['{col_lower}'].fillna('').str.strip() {op} {val_out})"
        else:
            # 多值型逻辑保持不变
            if op in ['==', '=']:
                return f"(df['{col_lower}'].fillna('').apply(lambda x: {val_out} in str(x)))"
            elif op in ['!=', '<>']:
                return f"(df['{col_lower}'].fillna('').apply(lambda x: {val_out} not in str(x)))"
            else:
                return f"(df['{col_lower}'].fillna('').apply(lambda x: str(x) {op} {val_out}))"
            
    
    # 解析WHERE表达式
    if 'W' in SQL:
        where = re.search(r'WHERE (.*)\s+GROUP\s+BY', sql, re.I).group(1)
        where = where.replace('AND', '&').replace('OR', '|')

        # 支持多种操作符的pattern
        pattern = re.compile(
            r"(\w+)\s*(==|=|!=|<>)\s*'([^']*)'"             # 1-3: 带引号字符串
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*(\d{4}/\d{1,2}/\d{1,2})" # 4-6: 裸日期
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)" # 7-9: 纯数字
        )

        where_py = pattern.sub(sql_condition_replace, where)
        where_py = re.sub(r"(?<!['\w])(\d{4}/\d{1,2}/\d{1,2})(?!['\w])", r"'\1'", where_py)
        # print('Python where:', where_py)
        
        cond = eval(where_py)
        for key in agg_funcs.keys():
            gt = get_result(df[cond], agg_funcs[key], group_col, key)

    else:
        for key in agg_funcs.keys():
            gt = get_result(df, agg_funcs[key], group_col, key)
    return gt, group_col

# %%

def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def compute_f1(gt, df):
    columns = gt.columns.tolist()
    gt_dict = dict(zip(gt[columns[0]], gt[columns[1]]))
    df_dict = dict(zip(df[columns[0]], df[columns[1]]))
    count = 0
    for key in gt_dict.keys():
        if key not in df_dict:
            count += 0
        else:
            if gt_dict[key] == df_dict[key]:
                count += 1
    p = count/len(df)
    r = count/len(gt)
    f1 = f1_score(p, r)
    return p, r, f1


#%%
def get_path():
    gt_in_path = "" # gt大表地址
    gt_out_path = "" # gt做完query保存的结果
    pre_path = "" # 各方法的地址
    prf1_path = "" # 最终的结果
    attr_types_path = "" # 属性的类型
    sql_path = "" # sql的地址
    return gt_in_path, gt_out_path, pre_path, prf1_path, attr_types_path, sql_path

# 读取sql
gt_in_path, gt_out_path, pre_path, prf1_path, attr_types_path, sql_path= get_path()
SQL = "" #['SFAG', 'SFWGA']
file = "" #['Wiki_Text', 'legal_case', 'player']
dic = {"p":[],"r":[],"f1":[]}

with open(sql_path, 'r', encoding='utf-8') as f:
    content = f.read()
sql_blocks = content.split('--------------------------------------------------')
first_10_sql = [sql_blocks[i].split('\n\n')[-1] for i in range(len(sql_blocks)-1)]

with open(attr_types_path, 'r', encoding='utf-8') as f:
    attr_types = json.load(f)[file]

for i in range(len(sql_blocks)-1):
    sql = first_10_sql[i]
    print(sql)
    df = pd.read_csv(pre_path)
    df_gt = pd.read_csv(gt_in_path)
    if file == 'Wikiart':
        df_gt = df_gt.head(200)
    if file == 'legal_case':
        df_gt = df_gt.head(100)
    if file == 'finance':
        df_gt = df_gt.head(30)
    gt, group_col = get_gt(sql, attr_types, df_gt, SQL)
    print(gt)
    gt.to_csv(gt_out_path, index=False)
    p, r, f1 = compute_f1(gt, df, attr_types, group_col)

    print(f"precision:{p}, recall:{r}, f1:{f1}")
    dic['p'].append(p)
    dic['r'].append(r)
    dic['f1'].append(f1)
df_final=pd.DataFrame(dic)
df_final.to_csv(prf1_path, index=False)

# %%
