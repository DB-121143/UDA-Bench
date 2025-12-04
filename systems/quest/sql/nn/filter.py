from .base import Physical

class Filter(Physical):
    """
    Info : columns - a list of ColumnExpr
            table - the retrieve tableName
            type - the retrieve type ('Photo'/'Text'/..)
            root - filter tree root
    Input : None
    """
    def __init__(self, columns, table, type, root):
        super().__init__()
        self.columns = columns
        self.table = table
        self.type = type
        self.root = root
        self.querier = None
        self.name = 'Filter'

    def process(self):
        pass

    def set_querier(self, x):
        self.querier = x