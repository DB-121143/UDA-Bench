from abc import ABCMeta, abstractmethod
from decimal import Decimal
from quest.conf import sqlconst


class ASTNode(object):
    """
    A ASTNode should includes:
    name : identifier the node type
    use var() to visit all attr
    if attr is type ASTNode then visit
    """
    __metaclass__ = ABCMeta

### Values BEGIN ###
# Value must be a leaf, return only a value

class ValueExpr(ASTNode):
    def __init__(self):
       self.value = None
       pass

    def parse_full(self):
        return self.value

class StringValue(ValueExpr):
    """
    'value'
    "value"
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'StringValue'
        self.value = '\'' + tokens[1:-1] + '\''

class IntegerValue(ValueExpr):
    def __init__(self, tokens):
        super().__init__()
        self.name = 'IntegerValue'
        self.value = int(tokens)

class RealValue(ValueExpr):
    """
    1.2
    -1.23
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'RealValue'
        self.value = float(tokens)

class ListValue(ValueExpr):
    """
    [x,y,..]
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'ListValue'
        self.value = []
        for v in tokens:
            self.value.append(v.value)

class RangeValue(ValueExpr):
    """
    [x,y]
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'RangeValue'
        self.value = list(tokens)

### Identifier Begin ###
# Identifiers must be a leaf, return only a value
# always return a list, but build in __init__

class IdentifierExpr(ASTNode):
    def __init__(self):
        pass

class ColumnExpr(IdentifierExpr):
    """
    column = [table_name, column_name, new_column_name]
    """
    def __init__(self, tokens):
        self.name = 'Column'
        self.column = tokens

    def parse(self):
        # only have column name here
        return self.column[1]
    
    def parse_column(self):
        # only have column name here
        return self.column[1]

    def parse_table(self):
        return self.column[0]
    
    def parse_full(self):
        if self.column[0] == sqlconst.DEFAULT_TABLE_NAME:
            return self.column[1]
        return self.column[0] + '.' + self.column[1]

class ColumnsExpr(IdentifierExpr):
    """
    a list with columns
    [[x,y,z], ...]
    x = table_name, y = column_name, z = new_column_name 
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Columns'
        self.columns = tokens

    def parse(self):
        res = []
        for col in self.columns:
            res.append(col[1])
        return res

class TablesExpr(IdentifierExpr):
    """
    a table attr only a name
    so build a set (x, y, z, ...)
    store fake name (now name)
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Tables'
        self.tables = tokens

### Binary ###
# returns are in tokens[0] as list [ls, op, rs]

class BinaryOperationExpr(ASTNode):
    """
    >, <, >=, <=, <>, =, AND, OR
    Binary get two children
    at list tokens [ls, op, rs]
    """
    def __init__(self, tokens):
        self.name = 'Binary'
        self.lhs, self.op, self.rhs = tokens

### PredicateExpr ###
# use for Expr that can be used for filtering, such as in a SELECT, WHERE, ON clause, HAVING clause
# All key words should be returned as name
# expr have different parts all in child

class PredicateExpr(ASTNode):
    def __init__(self):
        pass    

class SelectExpr(PredicateExpr):
    """
    [DISTINCT] : bool
    SELECT CLAUSE : AttrExpr
    FROM CLAUSE : FromExpr
    JOIN CLAUSE : JoinExpr
    WHERE CLAUSE : WhereExpr
    GROUP BY CLAUSE : GroupByExpr
    ORDER BY CLAUSE : OrderByExpr
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Select'
        self.distinct = tokens[0]
        self.selectClause = tokens[1]
        self.fromClause = tokens[2]
        self.joinClause = tokens[3]
        self.whereClause = tokens[4]
        self.groupbyClause = tokens[5]
        self.orderbyClause = tokens[6]

class AttrExpr(PredicateExpr):
    """
    a list includes columns/function
    or a special STAR
    e.g. [columnExpr, funtion, ...]
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Attribute'
        self.value = tokens[0]

class FromExpr(PredicateExpr):
    """
    TABLES (as a list [x, y, z])
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'From'
        self.value = tokens[0]

class WhereExpr(PredicateExpr):
    """
    value = conditions, a binary tree, e.g
            AND
        >          <
    A.T   10  B.R     100
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Where'
        self.value = tokens[0]

class HavingExpr(PredicateExpr):
    """
    a list of BinaryOperationExpr
    value = [condition]
    lhs = funtion, rhs = value, op
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Having'
        self.value = tokens[0]

class OrderByExpr(PredicateExpr):
    """
    ColumnsExpr.value = [table_name, column_name, new_column_name] 
    value = [[ColumnsExpr1, Ordered1(String)], ...]
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Order'
        self.value = tokens[0]

class JoinExpr(PredicateExpr):
    """
    JOIN = [JOIN TYPE (INNER/LEFT), TABLE, CONDITION (a binary op)]
    value = [JOIN1, JOIN2, ...]
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Join'
        self.value = tokens[0]

class GroupByExpr(PredicateExpr):
    """
    ColumnsExpr = [table_name, column_name, new_column_name]
    value = [ColumnsExpr1, ...]
    havingClause = havingExpr
    """
    def __init__(self, tokens):
        super().__init__()
        self.name = 'Group'
        #print("group by tokens:", tokens[0])
        self.value = tokens[0]
        self.havingClause = tokens[1]

### AggrExpr ###
# use for aggregate functions
# MIN, MAX, AVG, ,SUM, COUNT, etc.
# child will be ColumnsExpr

class FunctionExpr(ASTNode):
    def __init__(self, tokens):
        self.op = tokens[0] # string 
        self.column = tokens[1] # ColumnExpr

    def parse_column(self):
        return self.column.parse_column()
    
    def parse_table(self):
        return self.column.parse_table()
    
    def parse_full(self):
        return self.column.parse_full()

    def parsse(self):
        return self.column.parse()
    
    def parse_func(self):
        return self.op.lower() + '_' + self.parse_full()

class CountExpr(FunctionExpr):
    def __init__(self, tokens):
        super().__init__(tokens) 
        self.distinct = tokens[2] 
