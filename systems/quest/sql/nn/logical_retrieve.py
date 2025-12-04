from .base import Logical

class LogicalRetrieve(Logical):
    """
    Info : columns - a list of ColumnExpr
            table - the retrieve tableName
            type - the retrieve type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, table, type):
        super().__init__()
        self.columns = columns
        self.table = table
        self.type = type
        self.name = 'LogicalRetrieve'