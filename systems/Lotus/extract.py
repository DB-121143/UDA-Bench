#%%
import json
with open('../benchmark/extractions.json', 'r', encoding='utf-8') as f:
    extractions = json.load(f)["finance"]

with open('../benchmark/descriptions.json', 'r', encoding='utf-8') as f:
    descriptions = json.load(f)["finance"]

with open('./examples.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)["finance"]

print(extractions[0], descriptions[0], examples[0])

with open('../benchmark/Query/finance/SF.sql', 'r', encoding='utf-8') as f:
    content = f.read()
sql_blocks = content.split('--------------------------------------------------')
first_10_sql = [sql_blocks[i].split('\n\n')[-1] for i in range(len(sql_blocks)-1)]


# %%

import re
def anlysis_sfw_sql(sql):
    select_col = re.search(r'SELECT (.+?)\s+FROM', sql, re.I).group(1).strip()
    select_cols = [c.strip().lower() for c in select_col.split(',')]
    select_indices = [extractions.index(col.lower()) for col in select_cols if col.lower() in extractions]

    return select_indices



#%%
import pandas as pd

import lotus
from lotus.models import LM
import os
import numpy as np

os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""


csv_path = "../benchmark/ground_truth/finance.csv"
base_dir = "../benchmark/datasets/finance"

df = pd.read_csv(csv_path)
ids = df["ID"].dropna().astype(str).tolist()[:30]
# ids = [int(i.split('.txt')[0]) for i in os.listdir(base_dir)]

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
df = pd.DataFrame(data)
print(df.iloc[0])

# %%
all_indices = []
all_indices_sql = []
for j in range(len(sql_blocks)-1):
    folder = f'./results/finance/SF/SQL{j}'
    os.makedirs(folder, exist_ok=True)
    sql = first_10_sql[j]
    print(sql)
    select_indices = anlysis_sfw_sql(sql)
    print(select_indices)
    all_indices = list(set(all_indices) | set(select_indices))
    all_indices_sql.append(select_indices)

other_indices = [i for i in range(len(extractions)) if i not in all_indices]
print([extractions[i] for i in all_indices])

with open('extract/finance/all_indices_sql.json', 'w', encoding='utf-8') as f:
    json.dump(all_indices_sql, f, ensure_ascii=False, indent=2)
with open('extract/finance/all_indices.json', 'w', encoding='utf-8') as f:
    json.dump(all_indices, f, ensure_ascii=False, indent=2)

#%%
df_data = {"ID":[i for i in ids]}
for i in all_indices[-1:]:
    lm = LM(model="gpt-4.1-mini")
    lotus.settings.configure(lm=lm)
    att = extractions[i]
    description = descriptions[i]
    example = examples[i]
    # print(example)
    examples_df = pd.DataFrame(example)
    user_instruction = "What" + att + "in {context}?" + description + "If there are multiple values, separate them with '||' and leave empty if not applicable."
    df_test = df.sem_map(user_instruction, examples=examples_df)
    # df_test = df_filter.sem_map(user_instruction)
    df_test[['_map']].to_csv('extract/finance/result_'+att+'.csv', index=False, encoding='utf-8-sig')
    df_data[att] = df_test['_map'].tolist()
    print('-------')

# lm.print_total_usage()


# df_final = pd.DataFrame(df_data)
# df_final = df_final.map(lambda x: np.nan if isinstance(x, str) and "empty" in x else x)
# print(df_final.iloc[0])
# df_final.to_csv('extract/finance/result.csv', index=False, encoding='utf-8-sig')


# %%
