
class PrimaryKeyAligner:
    def __init__(self, primary_keys: PKSchema) -> None: ...
    
    def align(
        self,
        std_df: pd.DataFrame,
        pred_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        返回“标准、预测”两张 DataFrame，索引行严格对齐，并且排好序。如果出现重复的主键，则给出Warning并且随机保留一个。
        primary_keys的fuzzy=True 时，说明主键列是group by对应的列，先用e5 embedding block掉语义相似度太低的，再使用优先级匹配策略(先精确匹配，匹配不上交给LLM匹配)，注意使用LLM匹配不要一个个去调用api，要把需要用LLM检测的主键行添加到候选列表里，然后批量生成prompt，再让LLM去判断，根据判断结果决定这些行能否匹配。
        (比如NBA球员表，group by team,这时算准确率的主键列就变成了group by出来的这个team， pred_table中team是Boston Laker， std_table中team是 Hoston Lakers， 假设它们的语义相似度高于阈值，那就由LLM来判断这两个能不能匹配)
        异常抛 PrimaryKeyMismatchError。
        """

