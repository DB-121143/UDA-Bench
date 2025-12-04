from .base import Logical
from quest.core.node.logical_node import BinaryNode 

class LogicalFilter(Logical):
    """
    Info :  root - the root of filter tree, a FilterNode
            column - the columns need to be filter, related with table  -- table.column, columnExpr
            table - the filter tableName
    Input : LogicalRetrieve
    """
    def __init__(self, columns, table, root):
        super().__init__()
        self.columns = columns
        self.root = root
        self.table = table
        self.name = 'LogicalFilter'
        
