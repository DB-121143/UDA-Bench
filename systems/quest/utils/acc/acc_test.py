#%%
import pandas as pd
import re
import os
from tqdm import tqdm
import json
from litellm import batch_completion
os.environ["OPENAI_API_BASE"] = "https://aihubmix.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b"

# os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
# os.environ["OPENAI_API_KEY"] = "sk-INTd72cf84bd477b70433a9102a318263d8fdd1838emPKLc"

def clean_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗DataFrame中的ID列，将其转换为整数（去除小数点），再转为字符串类型。
    如果ID列不存在，则不做处理。
    """
    if 'id' in df.columns:
        # 先转为float，再转为int，最后转为str，避免小数点
        df['id'] = df['id'].astype(str).str.replace(r'\.0+$', '', regex=True)
    return df


#%%

def clean_file_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中的 file_name 列改名为 ID，并去掉文件名后缀（如 .txt, .pdf, .md 等）。
    
    参数:
        df: 包含 file_name 列的 pandas.DataFrame
        
    返回:
        新的 pandas.DataFrame，列名为 ID，且值已去除后缀
    """
    # 重命名列
    df = df.rename(columns={'file_name': 'id'})
    
    # 定义后缀正则：匹配末尾的 .txt .pdf .md 等
    suffix_pattern = re.compile(r'\.(txt|pdf|md)$', flags=re.IGNORECASE)
    
    # 去掉后缀
    df['id'] = df['id'].astype(str).str.replace(suffix_pattern, '', regex=True)
    
    return df


def get_gt(sql, attr_types, i, df, folder, SQL, df1):
    # 解析SELECT列
    select_col = re.search(r'SELECT (.+?)\s+FROM', sql, re.I).group(1).strip()
    select_cols = [c.strip().lower() for c in select_col.split(',')]
    for col in select_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0)
    out_cols = ['id'] + select_cols
            

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
        where = re.search(r'WHERE (.*);', sql, re.I).group(1)
        where = where.replace('AND', '&').replace('OR', '|')

        # 支持多种操作符的pattern
        pattern = re.compile(
            r"(\w+)\s*(==|=|!=|<>)\s*'([^']*)'"             # 1-3: 带引号字符串
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*(\d{4}/\d{1,2}/\d{1,2})" # 4-6: 裸日期
            r"|(\w+)\s*(==|=|!=|<>|>=|<=|>|<)\s*([0-9]+(?:\.[0-9]+)?)" # 7-9: 纯数字
        )

        where_py = pattern.sub(sql_condition_replace, where)
        where_py = re.sub(r"(?<!['\w])(\d{4}/\d{1,2}/\d{1,2})(?!['\w])", r"'\1'", where_py)
        print('Python where:', where_py)
        
        cond = eval(where_py)
        gt = df.loc[cond, out_cols]

    else:
        gt = df[out_cols]
    gt.to_csv(os.path.join(folder, 'filtered_gt.csv'),  index=False)
    return gt, select_cols, df1



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
        try:
            str_a = str(float(a)).strip() if not pd.isna(a) else ""
            str_b = str(float(b)).strip() if not pd.isna(b) else ""
        except:
            str_a = str(a).strip() if not pd.isna(a) else ""
            str_b = str(b).strip() if not pd.isna(b) else ""
        a_split = safe_split(str_a, sep_a) # 考虑到要测的属性可能是多值属性，多个值之间用||分隔，而且可能都是不能精确字符匹配的值。
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
                    - if number in List, If there are numerical values, a small margin of error within 1% is allowed.
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
                model="gpt-4.1-nano",
                messages=prompts[i:i+batch_size],
                stop=None,
                max_tokens=32,
                temperature=0,
            )

            for response in responses:
                content = response['choices'][0]['message']['content'].strip()
                results.append(int(content))

    print(f"匹配情况：{results}")

    p = sum([1 if len_a[i]==0 and len_b[i]==0 else min(results[i], len_a[i])/len_a[i] if len_a[i]!=0 else 0 for i in range(len(results))])/length_a
    r = sum([1 if len_a[i]==0 and len_b[i]==0 else min(results[i], len_b[i])/len_b[i] if len_b[i]!=0 else 0 for i in range(len(results))])/length_b


    return p, r, f1_score(p, r)


def compute_f1(gt, df, attr_types, select_cols, i, folder):
    p = r = f1 = 0
    gt = clean_id_column(gt)
    ids_intersection = set(df['id']) & set(gt['id'])
    matched_df_sub = df[df['id'].isin(ids_intersection)].copy()
    matched_gt_sub = gt[gt['id'].isin(ids_intersection)].copy()

    matched_df_sub = matched_df_sub.sort_values(by='id')
    matched_gt_sub = matched_gt_sub.sort_values(by='id')
    matched_df_sub = matched_df_sub.reset_index(drop=True)
    matched_gt_sub = matched_gt_sub.reset_index(drop=True)
    matched_df_sub.to_csv( os.path.join(folder, "matched_df_sub.csv"),  index=False)
    matched_gt_sub.to_csv( os.path.join(folder, "matched_gt_sub.csv"), index=False)
    print(f"pred table:\n{matched_df_sub}\n\nground truth table:\n{matched_gt_sub}")

    print(f"预测表格长度{len(df)}, gt长度{len(gt)}, 可作用表格{len(matched_df_sub)}")

    if len(matched_df_sub) == 0 or len(matched_gt_sub) == 0:
        print('预测表格或gt表格为空，直接返回0')
        return 0.0, 0.0, 0.0
    dic = {"p":[],"r":[],"f1":[], "col":[]}
    for col in select_cols:
        fixed = attr_types[col][1]
        p_col, r_col, f1_col = compute_col_f1(matched_df_sub[col], matched_gt_sub[col], len(df), len(gt), fixed)
        p += p_col
        r += r_col
        f1 += f1_col
        dic['p'].append(p_col)
        dic['r'].append(r_col)
        dic['f1'].append(f1_col)
        dic['col'].append(col)
    col_acc_df_final = pd.DataFrame(dic)
    col_acc_df_final.to_csv( os.path.join(folder, "col_acc_df_final.csv"), index=False)
    return p/len(select_cols), r/len(select_cols), f1/len(select_cols)

#%%

"""
结果文件夹结构：
```
legal_case/
	SF1.csv
	SF2.csv
	SFW1.csv
	SFW2.csv
	results/
		SF1/
			filtered_gt.csv
			matched_df_sub.csv
			matched_gt_sub.csv
			col_acc_df_final.csv
			avg_acc_df_final.csv
		SFW2/
			filtered_gt.csv
			matched_df_sub.csv
			matched_gt_sub.csv
			col_acc_df_final.csv
			avg_acc_df_final.csv			
```	
"""


def lowercase_columns(df_gt, df):
    """
    将两个pandas表格的列名全部改成小写。
    
    参数:
        df_gt: 第一个pandas.DataFrame
        df: 第二个pandas.DataFrame
        
    返回:
        tuple: 返回列名改为小写后的两个DataFrame (df_gt, df)
    """
    df_gt.columns = df_gt.columns.str.lower()
    df.columns = df.columns.str.lower()
    return df_gt, df

############################## 跑之前需要配置

# groundTruth、qurey相关根路径
ROOT_BENCHMARK_PATH = '/data/QUEST/benchmark'

# result_log根路径
ROOT_RESULT_LOG_PATH =  "/home/lijianhui/workspace/experiment_result/chunk/player/recursive"   #  "/data/QUEST/jzshe/project/quest/quest/tests/log" 

# /home/lijianhui/workspace/experiment_result/filter_optimize/player-8/quest-batch
# "/data/QUEST/jzshe/project/quest/quest/tests/log/zendb"

# ['Wiki_Text', 'legal_case', 'player', 'disease', 'drug', 'finance'] # 'disease'
table_to_test_acc = ['player']  # 'Wiki_Text', 'legal_case', 'player'
SQL_to_test =  ['SFW']  #  ['SF', 'SFW']

NEED_JUMP_EXISTING_RES = True # 当需要直接覆盖之前跑的错误结果时，设置这个flag.

# 我们只跑部分的实验
#####################
#######################
#######################
# head = 30 
head_dict ={
    'Wiki_Text': 200,
    'legal_case': 100,
    'player': 150, # 150 for quest, 50 for zendb ##########jznote
    'disease': 100,
    'drug': 100,
    'institutes': 100,
    'finance': 30
}
# 是否需要限制groundTruth表格长度, 当SQL类型为SF时，会自动缩减到extract_df的长度，若SQL类型为SFW, 需要人工按照下面的规则控制head

"""
nba全跑 142个
financial 30个
wikiart跑id前200个
其他的跑id前100个
"""



# table_name = None
# 读取sql
for SQL in SQL_to_test:
    for iter_table_name in table_to_test_acc:
        if SQL == 'SF':
            SF_SQL_PATH = os.path.join(ROOT_BENCHMARK_PATH, 'Query', iter_table_name ,'SELECT_FROM.sql')
            with open(SF_SQL_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            folder = os.path.join(ROOT_RESULT_LOG_PATH, iter_table_name) 
        else: ## SQL == "SFW"
            SFW_SQL_PATH = os.path.join(ROOT_BENCHMARK_PATH, 'Query', iter_table_name ,'SELECT_FROM_WHERE.sql')
            with open(SFW_SQL_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
            folder = os.path.join(ROOT_RESULT_LOG_PATH, iter_table_name) 
        sql_blocks = content.split('--------------------------------------------------')
        first_10_sql = [sql_blocks[i].split('\n\n')[-1] for i in range(len(sql_blocks)-1)] # len(sql_blocks)-1

        with open(os.path.join(ROOT_BENCHMARK_PATH, 'attr_types_all_unfixed.json'), 'r', encoding='utf-8') as f:
            attr_types = json.load(f)[iter_table_name]

        # 将attr_types字典中的所有key转化为小写:
        # 将attr_types字典中的所有key转化为小写:
        attr_types = {k.lower(): v for k, v in attr_types.items()}

        table_name = iter_table_name
        for i in range(len(sql_blocks)-1):  # len(sql_blocks)-1
            sql = first_10_sql[i]
            print(sql)

            

            ## 读取extract结果.csv
            if SQL == 'SF':
                extract_file = os.path.join(folder, f'SF{i+1}.csv')
                result_folder = os.path.join(folder, 'results', f'SF{i+1}')
            else:  # SQL == 'SFW'
                extract_file = os.path.join(folder, f'SFW{i+1}.csv') #
                result_folder = os.path.join(folder, 'results', f'SFW{i+1}')
            
            # 改成测出来的结果文件已经存在才能跳过，结果文件是result_folder + col_acc_df_final.csv
            result_file = os.path.join(result_folder, "col_acc_df_final.csv")
            if NEED_JUMP_EXISTING_RES and os.path.exists(result_file):
                continue
            os.makedirs(result_folder, exist_ok=True)

            if not os.path.exists(extract_file):
                print(f"Warning:  extract file {extract_file} not exist\n --------------------------------------------------")
                continue
            
            df = pd.read_csv(extract_file)

            head = head_dict[table_name]

            df_gt = pd.read_csv(os.path.join(ROOT_BENCHMARK_PATH, 'ground_truth', table_name + ".csv")).head(head)
            df = clean_file_names(df)

            
            df_gt, df = lowercase_columns(df_gt, df)
            gt, select_cols, df = get_gt(sql, attr_types, i, df_gt, result_folder + '/', SQL, df)
            p, r, f1 = compute_f1(gt, df, attr_types, select_cols, i, result_folder + '/')

            print(f"precision:{p}, recall:{r}, f1:{f1}")
            dic = {}
            dic['p'] = [p]
            dic['r'] = [r]
            dic['f1'] = [f1]
            avg_acc_df_final=pd.DataFrame(dic)
            avg_acc_df_final.to_csv(os.path.join(result_folder, 'avg_acc_df_final.csv'), index=False)

# %%
