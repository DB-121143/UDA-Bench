from .base import Physical

class Group(Physical):
    """
    Info : columns - a list of ColumnExpr
            table - the extract tableName
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, table, N_clusters, type):
        super().__init__()
        self.name = 'Group'
        self.columns = columns
        self.table = table
        self.N_clusters = N_clusters
        self.querier = None
        self.type = type

    def process(self):
        pass

    def set_querier(self, x):
        self.querier = x