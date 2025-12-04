
#%%
import pandas as pd
import re
import os
from tqdm import tqdm
import json
from litellm import batch_completion

API_BASE = "https://aihubmix.com/v1"
API_KEY = "sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b"

os.environ["OPENAI_API_BASE"] = API_BASE  #  "https://aihubmix.com/v1"  # "https://api.gptsapi.net/v1" # 
os.environ["OPENAI_API_KEY"] =  API_KEY  #  sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b # "sk-1hP6d6b4bdf693f90163482e714e9d69c308e26d283TquxQ" # 



#%%

#%%


def filter_parse_gt(sql, attr_types, head, gt_df, has_where=False):    

    # 解析SELECT列
    select_col = re.search(r'SELECT (.+?)\s+FROM', sql, re.I).group(1).strip()
    select_cols = [c.strip().lower() for c in select_col.split(',')]

    ## where start
    if has_where:

        # 解析WHERE表达式
        where = re.search(r'WHERE (.*);', sql, re.I).group(1)
        where = where.replace('AND', '&').replace('OR', '|')

        # 提取所有条件中的字段名，把非数值的转成str，防止后面用.str报错
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*(==|=|!=|<>|>=|<=|>|<)", where):
            field = m.group(1).lower()
            if (field in gt_df.columns) and (pd.api.types.is_numeric_dtype(gt_df[field]) is False):
                gt_df[field] = gt_df[field].astype(str)

        # 支持多种操作符的pattern
        pattern = re.compile(
            r"(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*'([^']*)'"             # 1-3: 带引号字符串
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*(\d{4}/\d{1,2}/\d{1,2})" # 4-6: 裸日期
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)" # 7-9: 纯数字
        )

        def sql_condition_replace(match):
            if match.group(1):  # 带引号字符串
                col = match.group(1)
                op = match.group(2)
                val = match.group(3)
                col_lower = col.lower()
                attr_type = attr_types.get(col_lower, ['single', 'fixed'])
                val_out = f"'{val}'"
            elif match.group(4):  # 裸日期
                col = match.group(4)
                op = match.group(5)
                val = match.group(6)
                col_lower = col.lower()
                attr_type = attr_types.get(col_lower, ['single', 'fixed'])
                return f"(df['{col_lower}'] {op} {val})"
            elif match.group(7):  # 纯数字
                col = match.group(7)
                op = match.group(8)
                val = match.group(9)
                col_lower = col.lower()
                attr_type = attr_types.get(col_lower, ['single', 'fixed'])
                val_out = f"'{val}'"
            else:
                raise ValueError(f"未知的SQL条件匹配：{match.groups()}")

            # 单值字符串与日期都走这里
            if attr_type[0] == 'single':
                if op in ['==', '=']:
                    return f"(df['{col_lower}'].fillna('').str.strip() == {val_out})"
                elif op in ['!=', '<>']:
                    return f"(df['{col_lower}'].fillna('').str.strip() != {val_out})"
                else:
                    return f"(df['{col_lower}'].fillna('').str.strip() {op} {val_out})"
            else:
                if op in ['==', '=']:
                    return f"(df['{col_lower}'].fillna('').apply(lambda x: {val_out} in str(x)))"
                elif op in ['!=', '<>']:
                    return f"(df['{col_lower}'].fillna('').apply(lambda x: {val_out} not in str(x)))"
                else:
                    return f"(df['{col_lower}'].fillna('').apply(lambda x: str(x) {op} {val_out}))"

        where_py = pattern.sub(sql_condition_replace, where)
        where_py = re.sub(r"(?<!['\w])(\d{4}/\d{1,2}/\d{1,2})(?!['\w])", r"'\1'", where_py)
        print('Python where:', where_py)

        cond = eval(where_py)

    ## where end

    # 默认ID字段存在且为大写，和表内列名保持一致
    out_cols = ['ID'] + select_cols
    # os.makedirs(f'/home/lijianhui/workspace/quest/log/cal_accuracy/SQL{i}', exist_ok=True)
    if has_where:
        filtered_gt = gt_df.loc[cond, out_cols]
    else:
        filtered_gt = gt_df[out_cols]
    # filtered_gt.to_csv(f'/home/lijianhui/workspace/quest/log/cal_accuracy/SQL{i}/gt.csv', index=False)
    return filtered_gt, select_cols

# %%

def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

#切割单元格
def safe_split(cell, sep = '||'):
    # 防止空值
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [s.strip() for s in str(cell).split(sep)]

def jaccard(a, b):
    a = [i.lower() for i in a]
    b = [i.lower() for i in b]
    a, b = set(a), set(b)
    if not a:
        return 0.0
    return len(a & b)

# 计算列的precision，recall，f1
def compute_col_f1(col_a, col_b, length_a, length_b, fixed, sep_a='||', sep_b='||', batch_size = 1):
    if length_a == 0 or length_b == 0:
        return 0.0, 0.0, 0.0
    prompts = []
    results = []
    len_a = []
    len_b = []
    for a, b in zip(col_a, col_b):
        str_a = str(a).strip() if not pd.isna(a) else ""
        str_b = str(b).strip() if not pd.isna(b) else ""
        a_split = safe_split(str_a, sep_a)
        b_split = safe_split(str_b, sep_b)
        len_a.append(len(a_split))
        len_b.append(len(b_split))

        # 固定列
        if fixed == 'fixed':
            results.append(jaccard(a_split, b_split))
        else:
            prompt = [
                {
                    "role": "system",
                    "content": "You are an expert in evaluating semantic similarity between terms. Respond strictly as instructed. Do not explain, infer, or expand. Begin your response immediately."
                },
                {
                    "role": "user",
                    "content": f"""Given two lists of terms, compare each term in List A with all terms in List B. For each term in List A, determine if there is any term in List B that expresses the same meaning (even if the words are different).
                    - Count how many terms in List A have at least one semantically similar match in List B.
                    - Respond with ONLY the final count (an integer), with nothing else.
                    - Do not explain, analyze, or add any reasoning.
                    - Begin your response immediately with the integer.

                    List A: {a_split}
                    List B: {b_split}
                    """
                }
            ]
            prompts.append(prompt)

    # 开放列
    if fixed == 'unfixed':
        results = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            responses = batch_completion(
                model="openai/gpt-4.1-mini",
                messages=prompts[i:i+batch_size],
                api_base=API_BASE,
                api_key=API_KEY,
                stop=None,
                max_tokens=32,
                temperature=0,
            )

            for response in responses:
                content = response['choices'][0]['message']['content'].strip()
                results.append(int(content))

    print(f"匹配情况：{results}")
    p = sum([results[i]/len_a[i] if len_a[i] != 0 else 0 for i in range(len(results))])/ length_a
    r = sum([results[i]/len_b[i] if len_b[i] != 0 else 0 for i in range(len(results))]) / length_b

    return p, r, f1_score(p, r)


def compute_f1(gt, df, attr_types, select_cols):
    p = r = f1 = 0
    df['ID'] = df['ID'].astype(str)
    gt['ID'] = gt['ID'].astype(str)
    ids_intersection = set(df['ID']) & set(gt['ID'])
    df_sub = df[df['ID'].isin(ids_intersection)].copy()
    gt_sub = gt[gt['ID'].isin(ids_intersection)].copy()

    df_sub = df_sub.sort_values(by='ID')
    gt_sub = gt_sub.sort_values(by='ID')
    df_sub = df_sub.reset_index(drop=True)
    gt_sub = gt_sub.reset_index(drop=True)
    # df_sub.to_csv('/home/lijianhui/workspace/quest/log/cal_accuracy/SQL'+str(i)+'/df_sub.csv', index=False)
    # gt_sub.to_csv('/home/lijianhui/workspace/quest/log/cal_accuracy/SQL'+str(i)+'/gt_sub.csv', index=False)
    print(f"pred table:\n{df_sub}\n\nground truth table:\n{gt_sub}")

    print(f"预测表格长度{len(df)}, gt长度{len(gt)}, 可作用表格{len(df_sub)}")

    if len(df_sub) == 0 or len(gt_sub) == 0:
        print('预测表格或gt表格为空，直接返回0')
        return 0.0, 0.0, 0.0, None, None

    accuracy_by_col = {

    }

    for col in select_cols:
        fixed = attr_types[col][1]
        print(f"整列{col}的匹配情况：")
        p_col, r_col, f1_col = compute_col_f1(df_sub[col], gt_sub[col], len(df), len(gt), fixed)
        accuracy_by_col[col] = {"p": 0.0, "r": 0.0, "f1": 0.0}
        accuracy_by_col[col]["p"] = p_col
        accuracy_by_col[col]["r"] = r_col
        accuracy_by_col[col]["f1"] = f1_col
        p += p_col
        r += r_col
        f1 += f1_col
    return p/len(select_cols), r/len(select_cols), f1/len(select_cols), gt_sub, df_sub, accuracy_by_col

#%%

def clean_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗DataFrame中的ID列，将其转换为整数（去除小数点），再转为字符串类型。
    如果ID列不存在，则不做处理。
    """
    if 'ID' in df.columns:
        # 先转为float，再转为int，最后转为str，避免小数点
        df['ID'] = df['ID'].astype(str).str.replace(r'\.0+$', '', regex=True)
    return df

def cal_accuracy(sql,  attr_types, gt, extract_df):    
    gt = clean_id_column(gt)
    # gt这个pd.DataFrame中的ID列可能是浮点数，需要清洗成整数
    filtered_gt, select_cols = filter_parse_gt(sql, attr_types,  1000, gt)
    p, r, f1,  gt_sub, df_sub, accuracy_by_col = compute_f1(filtered_gt, extract_df, attr_types, select_cols)
    return p, r, f1, filtered_gt, gt_sub, df_sub, accuracy_by_col # precision, recall, f1

import pandas as pd
import re


def clean_file_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中的 file_name 列改名为 ID，并去掉文件名后缀（如 .txt, .pdf, .md 等）。
    
    参数:
        df: 包含 file_name 列的 pandas.DataFrame
        
    返回:
        新的 pandas.DataFrame，列名为 ID，且值已去除后缀
    """
    # 重命名列
    df = df.rename(columns={'file_name': 'ID'})
    
    # 定义后缀正则：匹配末尾的 .txt .pdf .md 等
    suffix_pattern = re.compile(r'\.(txt|pdf|md)$', flags=re.IGNORECASE)
    
    # 去掉后缀
    df['ID'] = df['ID'].astype(str).str.replace(suffix_pattern, '', regex=True)
    
    return df


def extract_select_statements_from_sql_file(filepath):
    select_statements = []
    inside_select = False
    current_select = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 忽略空行和注释行
            stripped = line.strip()
            if not stripped or stripped.startswith('--'):
                continue

            # 跳过 CREATE TABLE
            if stripped.upper().startswith("CREATE TABLE"):
                inside_select = False
                current_select = []
                continue

            # 检测 SELECT 开始
            if not inside_select and stripped.upper().startswith("SELECT"):
                inside_select = True
                current_select.append(line.rstrip())
                # 如果 SELECT 语句直接以分号结尾，则立即存储
                if stripped.endswith(';'):
                    select_statements.append('\n'.join(current_select))
                    inside_select = False
                    current_select = []
                continue

            # 如果已在 SELECT 语句内部，继续收集直到遇到分号
            if inside_select:
                current_select.append(line.rstrip())
                if ';' in stripped:
                    select_statements.append('\n'.join(current_select))
                    inside_select = False
                    current_select = []
    return select_statements

def save_2_log(log_str, log_path = None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_str + '\n')
    return 

# 顶层调用
def cal_nba_players():

    nba_log_save_dir = "/home/lijianhui/workspace/quest/quest/tests/log/player/SF_accuracy.log"

    type = "SF"
    head = 1000
    num_sql = 10
    schema_root_path = "/data/QUEST/benchmark"
    # 当前要跑的测试集合 参数：
    table_file_name = "player.csv"    
    gt_df = pd.read_csv(os.path.join('/data/QUEST/benchmark/ground_truth', table_file_name)).head(head)
    sql_path = "/data/QUEST/jzshe/project/quest/data/benchmark/Query/player/SELECT_FROM.sql"
    sqls = extract_select_statements_from_sql_file(sql_path)
    with open(os.path.join(schema_root_path, 'attr_types_all_fixed.json') , 'r', encoding='utf-8') as f:
        attr_types = json.load(f)["player"] 
        # 把attr_types中的所有key转化成小写的 #todo
        attr_types = {k.lower(): v for k, v in attr_types.items()}
# ...existing code...
    

    sql_result_str = ""
    for i in range(num_sql):
        ext_csv_path = os.path.join('/home/lijianhui/workspace/quest/quest/tests/log/player', f"SF{i+1}.csv")

        if not os.path.exists(ext_csv_path):
            warning_str = f"{ext_csv_path} not exists"
            print(warning_str)
            save_2_log(warning_str, nba_log_save_dir)
            continue
        ext_df = pd.read_csv(ext_csv_path)
        ext_df = clean_file_names(ext_df)

        p, r, f1, filtered_gt, gt_sub, df_sub, accuracy_by_col  = cal_accuracy(sqls[i],  attr_types, gt_df, ext_df)
        os.makedirs(f"/home/lijianhui/workspace/quest/quest/tests/log/player/SF{i+1}", exist_ok=True)
        filtered_gt.to_csv(f"/home/lijianhui/workspace/quest/quest/tests/log/player/SF{i+1}/filtered_gt.csv", index=False)
        gt_sub.to_csv(f"/home/lijianhui/workspace/quest/quest/tests/log/player/SF{i+1}/gt_sub.csv", index=False)
        df_sub.to_csv(f"/home/lijianhui/workspace/quest/quest/tests/log/player/SF{i+1}/df_sub.csv", index=False)
        # accuracy_by_col 以json的形式存到本地。
        with open(f"/home/lijianhui/workspace/quest/quest/tests/log/player/SF{i+1}/accuracy_by_col.json", 'w', encoding='utf-8') as f:
            json.dump(accuracy_by_col, f, ensure_ascii=False, indent=4)
        

        
        output_str = f"SQL{i+1}: p={p}, r={r}, f1={f1}"
        print(output_str)
    
        save_2_log(output_str, nba_log_save_dir)
    return

if __name__ == "__main__":
    cal_nba_players()
