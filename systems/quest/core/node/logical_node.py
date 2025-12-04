
class FilterNode(object):
    """
    A filter tree (chain)
    type : now node type 'AND'/'OR'/'cmp'
    filterList : a list of FilterNode, if cmp then a BinaryNode
    """
    def __init__(self, type, table, filterList : list):
        self.type = type
        self.table = table
        self.filterList = filterList 
    
    def add_filter(self, filter):
        self.filterList.append(filter)

    def add_filter_list(self, lst):
        for v in lst:
            self.add_filter(v)

    def set_filterList(self, lst):
        self.filterList = lst

class BinaryNode(object):
    """
    use for binary operation in filter tree
    op = '>'/'<'/'>='/'<='/'=='/'IN'
    lhs = astn.columnExpr
    rhs = astn.valueExpr or astn.columnExpr (if 'IN')
    """
    def __init__(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def set_lhs(self, lhs):
        self.lhs = lhs

    def set_rhs(self, rhs):
        self.rhs = rhs
    
    def set_op(self, op):
        self.op = op

    def parse(self):
        return str(self.lhs.parse_full()) + ' ' + self.op + ' ' + str(self.rhs.parse_full())