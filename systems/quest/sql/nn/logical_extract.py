from .base import Logical

class LogicalExtract(Logical):
    """
    Info :  columns - the list of columnExpr
            table - the extract tableName
    Input : LogicalRetrieve / LogicalFilter
    """
    def __init__(self, columns, table):
        super().__init__()
        self.columns = columns
        self.table = table
        self.name = 'LogicalExtract'