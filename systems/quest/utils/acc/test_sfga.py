
import pandas as pd
import sys
sys.path.append("/data/QUEST/jzshe/project/quest")

from quest.utils.acc.mod_acc_test import cal_sql_acc


# sql = "SELECT position, AVG(age) FROM player GROUP BY position;"
# sql = "SELECT position, MAX(age) FROM player GROUP BY position;"
# sql = "SELECT position, MIN(age) FROM player GROUP BY position;"
sql = "SELECT position, COUNT(age) FROM player GROUP BY position;"


pred_table = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "position": ["Guards", "Forward", "RGuard", "Center", "HForward"],
    "avg(age)": [26.5, 29.5, 26.5, 30.0, 29.5],  # 故意包含计算错误
    "max(age)": [28, 32, 25, 30, 27],
    "min(age)": [25, 27, 25, 30, 27],
    "count(age)": [2, 2, 2, 1, 2]
})

gt_2_tables = {
    "player": pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "position": ["Guard", "Forward", "Guard", "Center", "Forward"],
        "age": [28, 32, 25, 30, 27],
        "draft_year": [2015, 2010, 2017, 2012, 2015],
        "fiba_world_cup": ["Spain", "USA", "France", "USA", "Australia"]
    })
}

acc_dict, filtered_gt_table = cal_sql_acc(sql, gt_2_tables, pred_table)
print(f"acc_dict: {acc_dict}")
print(f"filtered_gt_table: {filtered_gt_table}")


