#%%
import json
with open('../benchmark/extractions.json', 'r', encoding='utf-8') as f:
    extractions = json.load(f)["institutes"]

with open('../benchmark/descriptions.json', 'r', encoding='utf-8') as f:
    descriptions = json.load(f)["institutes"]

with open('./examples.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)["institutes"]

print(extractions[0], descriptions[0], examples[0])

with open('../benchmark/Query/institutes/SFW.sql', 'r', encoding='utf-8') as f:
    content = f.read()
sql_blocks = content.split('--------------------------------------------------')
first_10_sql = [sql_blocks[i].split('\n\n')[-1] for i in range(len(sql_blocks)-1)]


# %%

#sql
import re
def anlysis_sfw_sql(sql):
    select_col = re.search(r'SELECT (.+?)\s+FROM', sql, re.I).group(1).strip()
    select_cols = [c.strip().lower() for c in select_col.split(',')]
    select_indices = [extractions.index(col.lower()) for col in select_cols if col.lower() in extractions]

    where = re.search(r'WHERE (.*);', sql, re.I).group(1)

    attr_names = re.findall(
        r'([A-Za-z_][A-Za-z0-9_]*)\s*(?:==|=|!=|<>|>=|<=|>|<)', 
        where
    )
    attr_indices = [extractions.index(attr.lower()) for attr in attr_names if attr.lower() in extractions]

    return select_indices, where, attr_indices



#%%
import pandas as pd

import lotus
from lotus.types import CascadeArgs, ProxyModel
from lotus.models import LM
import os
import time
import numpy as np

os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


csv_path = "../benchmark/ground_truth/institutes.csv"
base_dir = "../benchmark/datasets/institutes"

df = pd.read_csv(csv_path)
ids = df["ID"].dropna().astype(str).tolist()

data = {'context':[]}
for id_value in ids:
    file_path = os.path.join(base_dir, f"{id_value}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            data['context'].append(content)
    else:
        data['context'].append('')

print(len(data['context']))


# %%
for j in range(len(sql_blocks)-1):
    lm = LM(model="gpt-4.1-mini")
    lotus.settings.configure(lm=lm)
    folder = f'./results/institutes/SFW/SQL{j}'
    os.makedirs(folder, exist_ok=True)
    sql = first_10_sql[j]
    select_indices, where, attr_indices = anlysis_sfw_sql(sql)
    print(select_indices, where, attr_indices)
    df = pd.DataFrame(data)

    user_instruction = "{context}."
    for i in attr_indices:
        user_instruction += descriptions[i]
    user_instruction += where
    
    filtered_df = df.sem_filter(
        user_instruction, strategy="Cot", return_all=True, return_explanations=False
    )
    print(filtered_df)
    print('#########')


    filtered_indices = filtered_df[filtered_df["filter_label"] == True].index
    contents = [data['context'][i] for i in filtered_indices]

    filter_data = {'context':contents}

    df_filter = pd.DataFrame(filter_data)

    df_data = {"ID":[ids[i] for i in filtered_indices]}
    for i in select_indices:
        att = extractions[i]
        description = descriptions[i]
        example = examples[i]
        # print(example)
        examples_df = pd.DataFrame(example)
        user_instruction = "What" + att + "in {context}?" + description + "If there are multiple values, separate them with '||' and leave empty if not applicable. Please keep each extracted value concise and avoid lengthy content."
        df_test = df_filter.sem_map(user_instruction, examples=examples_df)
        # df_test = df_filter.sem_map(user_instruction)
        df_data[att] = df_test['_map'].tolist()
    call = len(ids)+len(select_indices)*len(filtered_indices)
    print(f'LLM-call:{call}')
    print('-------')

    # lm.print_total_usage()
    

    df_final = pd.DataFrame(df_data)
    df_final = df_final.map(lambda x: np.nan if isinstance(x, str) and "empty" in x else x)
    # print(df_final.iloc[0])
    
    df_final.to_csv(folder+'/results.csv', index=False, encoding='utf-8-sig')

# %%
