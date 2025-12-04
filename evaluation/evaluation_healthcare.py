#%%
import pandas as pd
import re
import os
from tqdm import tqdm
import json
from litellm import batch_completion
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


#%%

def get_gt(sql, attr_types, i, df, folder, SQL, df1):
    # 解析SELECT列
    select_col = re.search(r'SELECT (.+?)\s+FROM', sql, re.I).group(1).strip()
    select_cols = [c.strip().lower() for c in select_col.split(',')]
    for col in select_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0)
    out_cols = ['ID'] + select_cols
            

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
    gt.to_csv(folder+'SQL'+str(i)+'/gt.csv', index=False)
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
def compute_col_f1(col_a, col_b, length_a, length_b, fixed, sep_a='||', sep_b='||', batch_size = 200):
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
    p = sum([1 if (len_a[i]==0 and len_b[i]==0) or (len_a[i]!=0 and min(results[i], len_a[i]) > 1)
    else (min(results[i], len_a[i])/len_a[i] if len_a[i]!=0 else 0)
    for i in range(len(results))])/length_a
    r = sum([1 if (len_a[i]==0 and len_b[i]==0) or (len_b[i]!=0 and min(results[i], len_b[i]) > 1)
    else (min(results[i], len_b[i])/len_b[i] if len_b[i]!=0 else 0)
    for i in range(len(results))])/length_b

    return p, r, f1_score(p, r)


def compute_f1(gt, df, attr_types, select_cols, i, folder):
    p = r = f1 = 0
    ids_intersection = set(df['ID']) & set(gt['ID'])
    df_sub = df[df['ID'].isin(ids_intersection)].copy()
    gt_sub = gt[gt['ID'].isin(ids_intersection)].copy()

    df_sub = df_sub.sort_values(by='ID')
    gt_sub = gt_sub.sort_values(by='ID')
    df_sub = df_sub.reset_index(drop=True)
    gt_sub = gt_sub.reset_index(drop=True)
    df_sub.to_csv(folder+'SQL'+str(i)+'/df_sub.csv', index=False)
    gt_sub.to_csv(folder+'SQL'+str(i)+'/gt_sub.csv', index=False)
    print(f"pred table:\n{df_sub}\n\nground truth table:\n{gt_sub}")

    print(f"预测表格长度{len(df)}, gt长度{len(gt)}, 可作用表格{len(df_sub)}")

    if len(df_sub) == 0 or len(gt_sub) == 0:
        print('预测表格或gt表格为空，直接返回0')
        return 0.0, 0.0, 0.0
    dic = {"p":[],"r":[],"f1":[], "col":[]}
    for col in select_cols:
        fixed = attr_types[col][1]
        p_col, r_col, f1_col = compute_col_f1(df_sub[col], gt_sub[col], len(df), len(gt), fixed)
        p += p_col
        r += r_col
        f1 += f1_col
        dic['p'].append(p_col)
        dic['r'].append(r_col)
        dic['f1'].append(f1_col)
        dic['col'].append(col)
    df_final = pd.DataFrame(dic)
    df_final.to_csv(folder+'SQL'+str(i)+'/results_col.csv', index=False)
    return p/len(select_cols), r/len(select_cols), f1/len(select_cols)

#%%

# 读取sql
for SQL in ['SFW','SF']:
    for file in ['disease', 'drug', 'institutes']:
        dic = {"p":[],"r":[],"f1":[]}
        if SQL == 'SF':
            with open('../benchmark/Query/'+file+'/SF.sql', 'r', encoding='utf-8') as f:
                content = f.read()
            folder = './results/'+file+'/SF/'
        else:
            with open('../benchmark/Query/'+file+'/SFW.sql', 'r', encoding='utf-8') as f:
                content = f.read()
            folder = './results/'+file+'/SFW/'
        sql_blocks = content.split('--------------------------------------------------')
        first_10_sql = [sql_blocks[i].split('\n\n')[-1] for i in range(len(sql_blocks)-1)]

        with open('attr_types.json', 'r', encoding='utf-8') as f:
            attr_types = json.load(f)[file]

        for i in range(len(sql_blocks)-1):
            sql = first_10_sql[i]
            print(sql)
            df = pd.read_csv(folder+'SQL'+str(i)+'/results.csv')
            df_gt = pd.read_csv('../benchmark/ground_truth/'+file+'.csv')
            if file == 'Wiki_Text':
                df_gt = df_gt.head(200)
            if file == 'Wiki_Image':
                df_gt = df_gt.head(200)
            if file == 'legal_case':
                df_gt = df_gt.head(100)
            if file == 'finance':
                df_gt = df_gt.head(30)
            gt, select_cols, df1 = get_gt(sql, attr_types, i, df_gt, folder, SQL, df)
            print(df1.head(20))
            print(gt.head(20))
            p, r, f1 = compute_f1(gt, df1, attr_types, select_cols, i, folder)

            print(f"precision:{p}, recall:{r}, f1:{f1}")
            dic['p'].append(p)
            dic['r'].append(r)
            dic['f1'].append(f1)
        df_final=pd.DataFrame(dic)
        df_final.to_csv(folder+'/results.csv', index=False)

# %%
