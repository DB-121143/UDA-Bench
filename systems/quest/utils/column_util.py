from quest.core.node import ast_node as astn
from quest.conf import sqlconst

def parse_full(x):
    if isinstance(x, list):
        res = []
        for v in x:
            res.append(v.parse_full())
        return res
    if isinstance(x, astn.ColumnExpr) or isinstance(x, astn.FunctionExpr):
        return x.parse_full()
    return x
    
def parse_column(x):
    if isinstance(x, list):
        res = []
        for v in x:
            res.append(v.parse_column())
        return res
    if isinstance(x, astn.ColumnExpr) or isinstance(x, astn.FunctionExpr):
        return x.parse_column()
    raise Exception('Can not parse_column')
    
def parse_table(x):
    if isinstance(x, list):
        res = []
        for v in x:
            res.append(v.parse_table())
        return res
    if isinstance(x, astn.ColumnExpr) or isinstance(x, astn.FunctionExpr):
        return x.parse_table()
    raise Exception('Can not parse_table')
    
def parse_column_and_func(x):
    if isinstance(x, list):
        res = []
        for v in x:
            res.append(parse_column_and_func(v))
        return res
    if isinstance(x, astn.FunctionExpr):
        return x.parse_func()
    if isinstance(x, astn.ColumnExpr):
        return x.parse_full()
    raise Exception('Can not parse_column_and_func')
    
def parse_func(x):
    if isinstance(x, list):
        res = []
        for v in x:
            res.append(v.parse_func())
        return res
    if isinstance(x, astn.FunctionExpr):
        return x.parse_func()
    raise Exception('Can not parse_func')

def parse_func_op(x):
    if isinstance(x, list):
        res = []
        for v in x:
            res.append(v.op)
        return res
    if isinstance(x, astn.FunctionExpr):
        return x.op
    raise Exception('Can not parse_func_op')