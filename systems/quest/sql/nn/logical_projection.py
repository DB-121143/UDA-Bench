from .base import Logical

class LogicalProjection(Logical):
    """
    Info :  columns - the list of columnExpr
    Input : LogicalFilter / LogicalJoin / LogicalExtract ..
    """
    def __init__(self, columns):
        super().__init__()
        self.columns = columns
        self.name = 'LogicalProjection'