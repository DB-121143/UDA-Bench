import os
import pandas as pd

base_dir = '/data/QUEST/jzshe/project/quest/quest/tests/log/drug/results'

# 遍历results目录下所有一级子目录
for sub_dir in os.listdir(base_dir):
    sub_dir_path = os.path.join(base_dir, sub_dir)
    if os.path.isdir(sub_dir_path):
        col_acc_path = os.path.join(sub_dir_path, 'col_acc_df_final.csv')
        avg_acc_path = os.path.join(sub_dir_path, 'avg_acc_df_final.csv')
        # 如果col_acc_df_final.csv存在且avg_acc_df_final.csv不存在，才处理
        if os.path.isfile(col_acc_path):  # and not os.path.isfile(avg_acc_path):
            # 读取csv
            df = pd.read_csv(col_acc_path, sep=',')
            # 计算p、r、f1平均值
            p_avg = df['p'].mean()
            r_avg = df['r'].mean()
            f1_avg = df['f1'].mean()
            # 构造平均结果的DataFrame
            avg_df = pd.DataFrame([{'p': p_avg, 'r': r_avg, 'f1': f1_avg}])
            # 保存为csv
            avg_df.to_csv(avg_acc_path, sep=',', index=False)
            print(f'写入完成: {avg_acc_path}')
