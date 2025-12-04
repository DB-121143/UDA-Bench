from .base import Physical

class Aggregation(Physical):
    """
    Info : functions - a list of FunctionExpr
            gp_columns - the list of groupby columnExpr
            columns - the list of columnExpr extracted before
            type - the extract type ('Photo'/'Text'/..)
    Input : table
    Output : table
    """
    def __init__(self, functions, gp_columns, columns, type):
        super().__init__()
        self.name = 'Aggregation'
        self.functions = functions
        self.gp_columns = gp_columns
        self.columns = columns
        self.type = type

    def process(self):
        pass