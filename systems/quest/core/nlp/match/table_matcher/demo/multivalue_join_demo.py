"""
表连接功能示例
展示如何使用表连接进行非精确匹配
"""

import pandas as pd
import os
import sys

sys.path.append("/data/QUEST/jzshe/project/quest")

from quest.core.nlp.match.table_matcher.table_join import pd_join_by_column, create_advanced_join_matcher, pd_join_by_column_with_join_type

from quest.core.nlp.match.fuse_join import  pd_fuse_join

from quest.core.nlp.match.table_matcher.modular_matcher import MatchingConfig,  create_llm_judge_matcher

from quest.conf.settings import JOIN_EDIT_DISTANCE_THRESHOLD, JOIN_SEMANTIC_THRESHOLD

DATA_ROOT_DIR = "/data/QUEST/jzshe/project/quest/log"
LEFT_TABLE_NAME = "drug.csv"
RIGHT_TABLE_NAME = "disease.csv"

def run_demo():
    """运行表连接示例"""
    print("加载示例数据...")
    
    t_left_table = pd.read_csv(os.path.join(DATA_ROOT_DIR, LEFT_TABLE_NAME)).head(20)
    t_right_table = pd.read_csv(os.path.join(DATA_ROOT_DIR, RIGHT_TABLE_NAME)).head(20) # jztodo
    
    print("\n原始左表:")
    print(t_left_table)
    
    print("\n原始右表:")
    print(t_right_table)


    # 使用不同的连接类型
    print("\n创建全面的匹配器并执行左连接...")
    edit_distance_matcher = create_advanced_join_matcher()
    
    
    inner_join_result = pd_fuse_join(
        t_left_table, t_right_table, 
        'disease_name', 'disease_name', 
        'TEXT',
        join_type='inner',
        matcher=edit_distance_matcher
    )
    
    print("\nfuse-inner-join结果:")
    print(inner_join_result)
    # 结果保存到本地：
    inner_join_result.to_csv(os.path.join(DATA_ROOT_DIR, 'test_fuse_join_result.csv'), index=False)


if __name__ == "__main__":
    run_demo()
