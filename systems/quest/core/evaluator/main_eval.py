

evaluator = ExtractionEvaluator()

gt_df = pd.read_csv(gt_table_path, encoding='utf-8')

sql="""SELECT player.name, team.name,  team.location
           FROM player
           INNER JOIN team ON player.team = team.name
        """

gt_cleaned_df = evaluator.clean(gt_df, primary_keys=["team_name"], type = "gt")
result_gt = evaluator.get_result_gt(sql, primary_keys=["team.id, player.id"], gt_table = gt_df) # primary_keys的列如果没有出现在sql中，需要加上

pred_df = pd.read_csv(pred_table_path, encoding='utf-8')

pred_cleaned_df = evaluator.clean(pred_df, type = "pred")
filtered_pred = evaluator.get_filtered_pred(sql, primary_keys=["team_name"], pred_table = pred_df)

# 单表join
report = evaluator.run(
    
    gt_table=df_player,      # 全量正确答案
    pred_table=df_pred,       # Agent 懒抽取得到
)
print(json.dumps(report, indent=2, ensure_ascii=False))
