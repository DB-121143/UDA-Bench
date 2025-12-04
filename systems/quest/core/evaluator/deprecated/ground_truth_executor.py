class GroundTruthSQLExecutor:
    """
    Orchestrator：一行代码跑评测
    """
    def __init__(
        self,
        primary_keys: PKSchema,
        delimiter: Delimiter = "||",
        semantic_cmp: Callable[[str, str], bool] | None = None,
    ): ...
    
    def run(
        self,
        sql: str,
        std_table: pd.DataFrame,
        pred_table: pd.DataFrame,
    ) -> EvalDict:
        """
        1. cleaner.clean 两表
        2. sql_executor.execute -> std_result_df
        3. aligner.align(std_result_df, pred_table)
        4. metrics.eval -> EvalDict
        """
