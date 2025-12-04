class DataCleaner:
    def __init__(self, delimiter: str = "||") -> None: ...
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - 列名统一小写 strip
        - object dtype → str
        - NaN, None ↔ 空串互转
        - 字符串值：trim 空格 (首尾空格要去掉，中间空白符只保留单个空格)
        - 多值列：去重 + trim 空格 
        """

    def clean_enum(self, df: pd.DataFrame, enum_cols_fields: dict[str, List[str]], enum_cols_fixed_type: dict[str, bool]) -> pd.DataFrame:
        """
        给出枚举列的列名和值域，将值映射到对应的枚举类型上
        enum_cols_fixed_type指出了对应的枚举列是否是固定类型
        """        
        pass
    
    def standardize_pk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - 主键列强制 str，去空格/大小写
        - 主键列去重，给出warning并且随机保留一行。
        """
