
import pandas as pd
import sys
sys.path.append("/data/QUEST/jzshe/project/quest")

from quest.utils.acc.mod_acc_test import cal_sql_acc


# 请根据下面的SQL语句为我写一个简单的测试抽取表格准确率的测试程序
# 改程序用于检测mod_acc_test.py是否支持对join形式的sql语句进行测试

sql = """
SELECT team.founded_year, city.population, city.state_name
FROM team
INNER JOIN city ON team.location = city.city_name;
"""

# pred_table应当具有下面的形式:
pred_table = pd.DataFrame({
    'team.founded_year': [1946, 1946, 1967],
    'city.population': [715522, 675647, 887642],
    'city.state_name': ['Colorado', 'Massachus', 'Indiana'],
    'team.file_name': ['1.txt', '5.txt', '3.txt'],
    'city.file_name': ['1.txt', '2.txt', '3.txt']
})


gt_city = pd.DataFrame(  {
    'city_name': ['Denver', 'Boston', 'Indianapolis'],
    'state_name': ['Colorado', 'Massachus', 'Indiana'],
    'population': [715522, 675647, 887642],
    'area': [400.739, 125.0, 953.0],
    'gdp': [None, None, None],  # 图片中gdp列为空
    'ID': [1, 2, 3]
} )

gt_team = pd.DataFrame(  {
    'team_name': ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets'],
    'founded_year': [1946, 1946, 1967],
    'location': ['Denver', 'Boston', 'Indianapolis'],
    'ownership': ['Atlanta Spirit LLC', None, 'Joseph Tsai'],  # 波士顿凯尔特人ownership为空
    'championships': [1, 18, 0],
    'ID': [1, 5, 3]
} )

gt_2_tables = {
    "team": gt_team,
    "city": gt_city
}


acc_dict, filtered_gt_table = cal_sql_acc(sql, gt_2_tables, pred_table)
print(f"acc_dict: {acc_dict}")
print(f"filtered_gt_table: {filtered_gt_table}")

