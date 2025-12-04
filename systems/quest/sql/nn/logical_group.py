from .base import Logical

class LogicalGroup(Logical):
    """
    Info :  columns - the list of columnExpr, grouped by these columns
            table - the extract tableName
    Input : LogicalRetrieve
    """
    def __init__(self, columns, table, N_clusters):
        super().__init__()
        self.columns = columns
        self.table = table
        self.N_clusters = N_clusters
        self.name = 'LogicalExtract'