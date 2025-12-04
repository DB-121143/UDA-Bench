import os
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import re
import sys
import litellm
#litellm._turn_on_debug()

sys.path.append('/data/QUEST/jzshe/project/quest')
from tqdm import tqdm

from quest.utils.acc.mod_acc_test_copy import cal_sql_acc, clean_pd
os.environ["OPENAI_API_BASE"] = "https://aihubmix.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b"



def parse_sql_file_strict(content):
    """
    更严格的解析方式，确保只提取SELECT语句
    """
    result = {}
    
    # 按分隔线分割
    sections = re.split(r'-{40,}', content)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        query_id = None
        select_lines = []
        in_select = False
        
        for line in lines:
            # 查找Query编号
            query_match = re.search(r'-- Query (\d+)', line)
            if query_match:
                query_id = int(query_match.group(1))
                continue
            
            # 检查是否是SELECT语句的开始
            if re.match(r'\s*SELECT\b', line, re.IGNORECASE):
                in_select = True
                select_lines = [line]
                continue
            
            # 如果在SELECT语句中，继续收集行
            if in_select:
                select_lines.append(line)
                # 如果遇到分号，SELECT语句结束
                if ';' in line:
                    break
        
        # 如果找到了query_id和SELECT语句
        if query_id is not None and select_lines:
            # 合并SELECT语句行
            select_sql = ' '.join(select_lines)
            # 清理格式
            select_sql = re.sub(r'\s+', ' ', select_sql.strip())
            # 移除结尾分号
            # select_sql = select_sql.rstrip(';')
            result[query_id] = select_sql
    
    return result


def read_sqls(sql_path: str) -> dict:
    with open(sql_path, 'r') as f:
        sql_str = f.read()
    result = parse_sql_file_strict(sql_str)    
    return result




def read_result_and_test():

    ###########################################核心自定义参数配置
    NEED_JUMP_EXISTING_RES = True
    # 只跑部分实验，用于缩减gt的长度
    head_dict ={
        'Wiki_Text': 200,
        'legal_case': 100,
        'player': 150, # 150 for quest, 50 for zendb ##########jznote
        'disease': 100,
        'drug': 100,
        'institutes': 100,
        'finance': 30
    }        
    sql_path = "/home/lijianhui/workspace/experiment_result/aggregation.sql"
    SQL_TAG = "SFWGA"    
    
    gt_2_path_dict = {
        'player':"/data/QUEST/benchmark/ground_truth/player.csv"
    }

    Result_Root_Path = "/home/lijianhui/workspace/experiment_result/sfwga/player/sen/result"
    Pred_Table_Root_Path = "/home/lijianhui/workspace/experiment_result/sfwga/player/sen"

    ###########################################核心自定义参数配置

    sqls = read_sqls(sql_path)
    gt_2_tables = {}
    for gt_path in gt_2_path_dict.items():
        gt_2_tables[gt_path[0]] = clean_pd(pd.read_csv(gt_path[1], encoding='utf-8'))
        gt_2_tables[gt_path[0]] = gt_2_tables[gt_path[0]].head(head_dict[gt_path[0]])

    for sql_id, sql in sqls.items():
        # 存储对应sql query的结果到csv文件: 还包括filtered_gt_table(SQL作用在GT上得到的结果)
        pred_table_path = os.path.join(Pred_Table_Root_Path, f"{SQL_TAG}{sql_id}.csv")
        result_table_dir =  os.path.join(Result_Root_Path,  f"{SQL_TAG}{sql_id}")
        acc_result_table_path = os.path.join(result_table_dir,  "col_acc_df_final.csv")     

        if NEED_JUMP_EXISTING_RES:
            if os.path.exists(acc_result_table_path):
                print(f"{SQL_TAG} sql_id: {sql_id}的结果已存在，跳过")
                continue
            else:
                os.makedirs(result_table_dir, exist_ok=True)

        if not os.path.exists(pred_table_path):
            print(f"Warning:  extract file {pred_table_path} not exist\n --------------------------------------------------")
            continue
        
        pred_table = clean_pd(pd.read_csv(pred_table_path, encoding='utf-8'))
        pred_table = pred_table.head(head_dict['player'])  

        acc_dict,filtered_gt_table = cal_sql_acc(sql, gt_2_tables, pred_table)
        # 分别存储acc_dict和filtered_gt_table到result_table_dir
        filtered_gt_table.to_csv(os.path.join(result_table_dir, "filtered_gt_table.csv"), index=False)
        acc_df = pd.DataFrame(acc_dict)
        acc_df.to_csv(acc_result_table_path, index=False)
        print(f"sql_id: {sql_id}的accuracy结果已存储到{acc_result_table_path}")
        print(acc_dict)

    return 


if __name__ == "__main__":
    read_result_and_test()
