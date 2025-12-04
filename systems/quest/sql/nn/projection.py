from .base import Physical

class Projection(Physical):
    """
    Info : columns - a list of ColumnExpr
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, type):
        super().__init__()
        self.name = 'Projection'
        self.columns = columns
        self.type = type
        self.indexer = None

    def set_indexer(self, x):
        self.indexer = x

    def process(self):
        pass