
class DuckDBExecutor:
    """
    负责把 pandas.DataFrame 注册到 DuckDB，执行 SQL 并返回结果 DataFrame。
    """
    def __init__(self, register_udf: bool = True) -> None: ...
    
    def execute(
        self,
        sql: str,
        tables: dict[str, pd.DataFrame],  # {"player": df1, "team": df2}
        **kwargs,

    ) -> pd.DataFrame:
        pass

    def sql_convert(self, sql: str, **kwargs) -> str:
        """
        先解析 SQL（用 sqlglot），然后转换为 DuckDB 的 SQL
        """
        pass

