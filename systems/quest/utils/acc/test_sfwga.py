
import pandas as pd
import sys
sys.path.append("/data/QUEST/jzshe/project/quest")
from quest.utils.acc.mod_acc_test import cal_sql_acc

sql = "SELECT team, COUNT(*) FROM player WHERE draft_year<=1979 OR olympic_gold_medals==3 GROUP BY team;"

# 新增测试用例1：包含draft_year<=1979和olympic_gold_medals==3的多种情况
gt_2_tables_1 = {
    "player": pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "team": ["A", "A", "B", "B", "C", "C", "A"],
        "draft_year": [1978, 1980, 1975, 1985, 1979, 1982, 1979],
        "olympic_gold_medals": [2, 3, 1, 3, 0, 3, 3]
    })
}
pred_table_1 = pd.DataFrame({
    "team": ["A", "B", "C"],
    "COUNT(*)": [3, 2, 2]
})

# 新增测试用例2：没有符合条件的行
gt_2_tables_2 = {
    "player": pd.DataFrame({
        "id": [1, 2, 3],
        "team": ["X", "Y", "Z"],
        "draft_year": [1980, 1981, 1982],
        "olympic_gold_medals": [1, 2, 0]
    })
}
pred_table_2 = pd.DataFrame({
    "team": [],
    "COUNT(*)": []
})

# 新增测试用例3：所有行都符合条件
gt_2_tables_3 = {
    "player": pd.DataFrame({
        "id": [1, 2, 3],
        "team": ["X", "X", "Y"],
        "draft_year": [1970, 1975, 1979],
        "olympic_gold_medals": [3, 3, 3]
    })
}
pred_table_3 = pd.DataFrame({
    "team": ["X", "Y"],
    "COUNT(*)": [2, 1]
})

# 新增测试用例4：部分行符合条件，team有重复
gt_2_tables_4 = {
    "player": pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "team": ["A", "A", "B", "B", "A"],
        "draft_year": [1978, 1985, 1979, 1980, 1979],
        "olympic_gold_medals": [1, 3, 0, 2, 3]
    })
}
pred_table_4 = pd.DataFrame({
    "team": ["A", "B"],
    "COUNT(*)": [3, 1]
})

# 打印每个测试用例的结果
for i, (gt, pred) in enumerate([
    (gt_2_tables_1, pred_table_1),
    (gt_2_tables_2, pred_table_2),
    (gt_2_tables_3, pred_table_3),
    (gt_2_tables_4, pred_table_4)
], 1):

    acc_dict, filtered_gt_table = cal_sql_acc(sql, gt, pred)
    print(f"Test case {i}:")
    print(f"acc_dict: {acc_dict}")
    print(f"filtered_gt_table:\n{filtered_gt_table}\n")



