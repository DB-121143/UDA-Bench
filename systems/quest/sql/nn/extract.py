from .base import Physical

class Extract(Physical):
    """
    Info : columns - a list of ColumnExpr
            table - the extract tableName
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, table, type):
        super().__init__()
        self.name = 'Extract'
        self.columns = columns
        self.table = table
        self.querier = None
        self.type = type

    def process(self):
        pass

    def set_querier(self, x):
        self.querier = x