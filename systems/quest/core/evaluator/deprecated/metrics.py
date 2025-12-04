class MetricsEvaluator:
    def __init__(
        self,
        delimiter: Delimiter = "||",
        semantic_cmp: Callable[[List[str], List[str]], bool] | None = None,
    ): ...
    
    def _split(self, cell: Any) -> set[str]:
        ...
    
    def _prf(self, gold: set[str], pred: set[str]) -> tuple[float, float, float]:
        """
        公式：
        \\[
        P = \\frac{|G \\cap P|}{|P|},\\;
        R = \\frac{|G \\cap P|}{|G|},\\;
        F1 = \\frac{2PR}{P+R}
        \\]
        """
        pass
    
    def eval(
        self,
        std_df: pd.DataFrame,
        pred_df: pd.DataFrame,
    ) -> EvalDict:
        """
        - 按列遍历 _split 比对
        - overall 先行级联 micro-averaged
        - semantic_cmp: 可注入比如 WordNet 同义词判等函数，对主键匹配的2列进行批量比对
        """
        pass
