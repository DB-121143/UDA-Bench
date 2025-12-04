from .base import Physical

class Retrieve(Physical):
    """
    Info : columns - a list of ColumnExpr
            table - the retrieve tableName
            type - the retrieve type ('Photo'/'Text'/..)
            retrieveList - the retrieve id
            indexer : a SingleIndexer to retrieve data
    Input : None
    """
    def __init__(self, columns, table, type):
        super().__init__()
        self.columns = columns
        self.table = table
        self.type = type
        self.retrieveList = []
        self.indexer = None 
        self.sampler = None
        self.name = 'Retrieve'

    def set_retrieveList(self, x):
        self.retrieveList = x

    def set_indexer(self, x):
        self.indexer = x

    def set_sampler(self, x):
        self.sampler = x