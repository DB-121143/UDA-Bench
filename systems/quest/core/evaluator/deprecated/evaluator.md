# 总体模块架构

|                          |                                        |                                                                                          |
| ------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------- |
| 模块                     | 责任                                   | 关键类 / 方法                                                                            |
| sql_executor.py          | 在 标准表 上安全地执行 SQL             | DuckDBExecutor.execute(sql: str, tables: dict[str, pd.DataFrame]) → pd.DataFrame        |
| cleaner.py               | 统一大小写、空值、拆分多值、主键规范化 | DataCleaner.clean(df), DataCleaner.standardize_pk(df, primary_keys)                         |
| aligner.py               | 主键级别对齐 + 模糊匹配策略            | PrimaryKeyAligner.align(std_df, pred_df, primary_keys) → tuple[pd.DataFrame, pd.DataFrame] |
| metrics.py               | 按列准确率 P, R, F1 评测              | `MetricsEvaluator.eval(std_df, pred_df, *, delimiter=",")                                |
| ground_truth_executor.py | 主流程 Orchestrator                    | GroundTruthSQLExecutor.run(sql, std_df, pred_df, primary_keys, …) → EvalReport            |
| udf.py (可选)            | 注册自定义 DuckDB UDF，用于复杂列操作  | 如 array_contains, normalize_str…                                                       |
| exceptions.py            | 语义化错误类型                         | SQLParseError, PrimaryKeyMismatchError, …                                               |
| typing.py                | TypedDict / NewType for 媒介类型       | EvalDict, PKSchema                                                                       |
