from .base import Logical

class LogicalAggregation(Logical):
    """
    Info :  functions - the list of functionExpr
            gp_columns -  the list of groupby columnExpr
            columns - the list of extracted columnExpr
            table - the extract tableName
    Input : Table

    only need to apply aggr functions to the table
    """
    def __init__(self, functions, gp_columns, columns):
        super().__init__()
        self.functions = functions
        self.gp_columns = gp_columns
        self.columns = columns
        self.name = 'LogicalAggregation'