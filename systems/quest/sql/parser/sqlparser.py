
import ply.lex as lex
import ply.yacc as yacc
from quest.core.node import ast_node as astn
from quest.conf import sqlconst

#import myast.ast_node as astn
#import myparser.consts as const


### LEX BEGIN ###

"""
TODO
    BETWEEN
    ADD/MINUS/..
"""

keywords = {
    'SELECT': 'SELECT',
    'FROM': 'FROM',
    'WHERE': 'WHERE',
    'ON': 'ON',
    'IN': 'IN',
    'AS': 'AS',
    'INNER': 'INNER',
    'JOIN': 'JOIN',
    'LEFT': 'LEFT',
    'GROUP': 'GROUP',
    'ORDER': 'ORDER',
    'BY' :'BY',
    'HAVING': 'HAVING',
    'MIN': 'MIN',
    'MAX': 'MAX',
    'COUNT': 'COUNT',
    'SUM': 'SUM',
    'AVG': 'AVG',
    'DISTINCT': 'DISTINCT',
    'AND': 'AND',
    'OR': 'OR',
    'ASC': 'ASC',
    'DESC': 'DESC'
}

# identifer and constant

tokens = [
    'IDENTIFIER',
    'QIDENTIFIER',
    'NUMBER',
    'LPAREN',  
    'RPAREN', 
    'COMA',  # ','
    'DOT',  # '.'
    'DQ',  # ' " '
    'EQ',  # '='
    'NEQ',  # '<>'
    'LT',  # '<'
    'LEQ',  # '<='
    'GT',  # '>'
    'GEQ',  # '>='
    'STAR' # '*'
] + list(keywords.values())

# grammars

t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMA = r','
t_DOT = r'\.'
t_DQ = r'"'
t_EQ = r'=='
t_NEQ = r'<>|!='
t_LT = r'<'
t_LEQ = r'<='
t_GT = r'>'
t_GEQ = r'>='
t_STAR = r'\*'

# ignores, careful may ignore key like 'long long' with space

t_ignore = ' \t'

# function grammars

def t_NUMBER(t):
    # only int now, can add others
    r'-?\d+(\.\d+)?'
    if '.' in t.value:
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = keywords.get(t.value, 'IDENTIFIER')
    return t

def t_QIDENTIFYER(t):
    r'[\'"]([^\'"]*)[\'"]'
    t.type = keywords.get(t.value, 'QIDENTIFIER')
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    return t

# errors
def t_error(t):
    raise Exception('Lex error {} at line {}, illegal character {}'
                    .format(t.value[0], t.lineno, t.value[0]))

### LEX END ###

### YACC BEGIN ###

# {key: value}, value is real name, key is a name you like, otable is oppo
# for example, T1 as x, in tables {T1: x} origin_tables {x: T1}
origin_tables = {}

# {key: value}, value is real name, key is a name you like, otable is oppo
# for example, T1.a as x, in tables {T1: x} origin_tables {x: T1.a}
origin_columns = {}

# grammars
# In grammars, AST build a node only when returned to top (if the return type is a list)
# for example, p_column return [[x,a]] p_columns return [[x,a], [y,b], ...] 
# the list show by reversed before build !!!!!!!!!!
# select_statment tables build ColumnsExpr()

# define the Combination and Precedence
# e.g. 
# ('left', 'PLUS', 'MINUS')
# ('left', 'TIMES', 'DIVIDE') 
# because TIMES and DIVIDE do first
precedence = (
    ('nonassoc', 'LT', 'LEQ', 'GT', 'GEQ', 'EQ', 'NEQ'),  # Nonassociative operators
    ('left', 'OR'),
    ('left', 'AND'), # AND regular first
)

def p_start(p):
    """
    start : statement
    """
    p[0] = p[1]
    
def p_empty(p):
    """
    empty :
    """
    p[0] = None
    pass

def p_select_statement(p):
    """
    select_statement : SELECT distinct attributes FROM tables joins where groups order_by
    """
    flag = True if p[2]=='DISTINCT' else False

    p[5].reverse()
    fromClause = astn.FromExpr([p[5]])
    p[3].reverse()
    attrClause = astn.AttrExpr([p[3]])
    # [DISTINCT], ATTRIBUTES, FROM CLAUSE, [JOIN CLAUSE], [WHERE CLAUSE], 
    # [GROUP BY CLAUSE], [ORDER BY CLAUSE]
    p[0] = astn.SelectExpr([flag, attrClause, fromClause, p[6], p[7], p[8], p[9]]) 
    # print("get attrs:", attrClause.child[0].column)

def p_statement(p):
    """
    statement : select_statement
    """
    p[0] = p[1]

def p_distinct(p):
    """
    distinct : DISTINCT 
             | empty
    """
    p[0] = p[1]

### here may add attribute
def p_attributes(p):
    # TODO 
    """
    attributes : column
               | function
               | column COMA attributes
               | function COMA attributes
               | STAR
    """
    pnow = None
    if p[1]=='*':
        pnow = astn.ColumnExpr(sqlconst.ALL_COLUMNS_STAR_LIST)
    elif isinstance(p[1], astn.FunctionExpr):
        pnow = p[1]
    else:
        pnow = p[1]
    
    if len(p)>2:
        p[3].append(pnow)
        p[0] = p[3]
    else:
        p[0] = [pnow]

    # print("attr:", p[0])


def p_column(p):
    # TODO
    # IDENTIFER with DOT not solved
    """
    column : IDENTIFIER DOT IDENTIFIER
           | IDENTIFIER DOT IDENTIFIER AS IDENTIFIER
           | IDENTIFIER
    """
    table_name = sqlconst.DEFAULT_TABLE_NAME
    column_name = sqlconst.DEFAULT_COLUMN_NAME
    new_column_name = sqlconst.DEFAULT_COLUMN_NAME
    if len(p)>2:
        table_name = p[1]
        column_name = p[3]
    else:
        column_name = p[1]

    # a bit dif with table
    # careful if add matching column : IDENTIFIER
    if len(p)>4:
        origin_columns.setdefault(p[5], [table_name, p[3]])
        new_column_name = p[5]
    
    p[0] = astn.ColumnExpr([table_name, column_name, new_column_name])

def p_columns(p):
    """
    columns : column
            | column COMA columns
    """
    if len(p)<3:
        p[0] = [p[1]]
    else:
        p[0] = p[4].append(p[1])


### careful with STAR, only COUNT(*) used here
# TODO update other aggrs
def p_function(p):
    """
    function : MIN LPAREN column RPAREN
             | MAX LPAREN column RPAREN
             | SUM LPAREN column RPAREN
             | AVG LPAREN column RPAREN
             | COUNT LPAREN column RPAREN
             | COUNT LPAREN DISTINCT column RPAREN
             | COUNT LPAREN STAR RPAREN
    """
    if p[1]=='COUNT':
        if p[3]=='*':
            p[0] = astn.CountExpr([p[1], astn.ColumnExpr(sqlconst.ALL_COLUMNS_STAR_LIST), False])
        elif p[3]=='DISTINCT':
            p[0] = astn.CountExpr([p[1], p[4], True])
        else:
            p[0] = astn.CountExpr([p[1], p[3], False])
    else:
        p[0] = astn.FunctionExpr([p[1],p[3]])

def p_joins(p):
    """
    joins : join
    """
    if isinstance(p[1], list):
        p[1].reverse()
        p[0] = astn.JoinExpr([p[1]])

def p_join(p):
    """
    join : INNER JOIN table ON condition join
         | LEFT JOIN table ON condition join
         | empty
    """
    if len(p)>2:
        if isinstance(p[6], list):
            p[6].append([p[1], p[3], p[5]])
            p[0] = p[6]
        else:
            p[0] = [[p[1], p[3], p[5]]]
    else:
        p[0] = None

def p_where(p):
    """
    where : WHERE conditions
          | empty
    """
    if len(p)>2:
        p[0] = astn.WhereExpr([p[2]])

def p_condition(p):
    # TODO column in select_statement
    # TODO column IN LPAREN select_statement RPAREN
    """
    condition : column operator value
              | column operator column
              | column IN LPAREN values RPAREN
    """
    rhs = None
    if len(p)>4:
        p[4].reverse()
        rhs = astn.ListValue(p[4])
    elif p[3] != '(':
        rhs = p[3]
    p[0] = astn.BinaryOperationExpr([p[1], p[2], rhs])


# here AND level higher than OR
# no PARER!
def p_conditions(p):
    """
    conditions : condition
               | conditions AND conditions
               | conditions OR conditions
               | LPAREN conditions RPAREN
    """
    if p[1]=='(':
        p[0] = p[2]
    elif len(p)>2:
        p[0] = astn.BinaryOperationExpr([p[1], p[2], p[3]])
    else: 
        p[0] = p[1]

def p_groups(p):
    """
    groups : group_by
           | group_by having
           | empty
    """
    # having should after group_by
    if len(p)==3:
        p[0] = astn.GroupByExpr([p[1], p[2]])
    elif isinstance(p[1], list):
        p[0] = astn.GroupByExpr([p[1], None])


def p_group_by(p):
    """
    group_by : GROUP BY columns
    """
    p[3].reverse()
    p[0] = p[3]

def p_having(p):
    """
    having : HAVING conditions_having
    """
    p[2].reverse()
    p[0] = astn.HavingExpr([p[2]])

def p_conditions_having(p):
    """
    conditions_having : function operator value
                      | function operator value COMA conditions_having
    """
    if len(p)<5:
        p[0] = [astn.BinaryOperationExpr([p[1], p[2], p[3]])]
    else:
        p[5].append(astn.BinaryOperationExpr([p[1], p[2], p[3]]))
        p[0] = p[5]

def p_order_by(p):
    """
    order_by : ORDER BY conditions_order_by
             | empty
    """
    if len(p)>2:
        p[3].reverse()
        p[0] = astn.OrderByExpr([p[3]])

def p_conditions_order_by(p):
    """
    conditions_order_by : column ordered
                        | column ordered COMA conditions_order_by
    """
    if len(p)<4:
        p[0] = [[p[1], p[2]]]
    else:
        p[4].append([p[1],p[2]])
        p[0] = p[4]

def p_ordered(p):
    """
    ordered : ASC
            | DESC
    """
    p[0] = p[1]

def p_tables(p):
    """
    tables : table
           | table COMA tables
    """
    if len(p)<3:
        p[0] = [p[1]]
    else:
        p[3].append(p[1])
        p[0] = p[3]

def p_table(p):
    """
    table : IDENTIFIER
          | IDENTIFIER AS IDENTIFIER
          | IDENTIFIER IDENTIFIER
    """
    if len(p) == 2:
        origin_tables.setdefault(p[1], p[1])
        p[0] = p[1]
    elif len(p) == 4:
        origin_tables.setdefault(p[3], p[1])
        p[0] = p[3]
    elif len(p) == 3:
        origin_tables.setdefault(p[2], p[1])
        p[0] = p[2]
    
def p_values(p):
    """
    values : value
           | value COMA values
    """
    if len(p)<3:
        p[0] = [p[1]]
    else:
        p[3].append(p[1])
        p[0] = p[3]

def p_value(p):
    """
    value : NUMBER
          | QIDENTIFIER
    """
    if isinstance(p[1], str):
        p[0] = astn.StringValue(p[1])
    elif isinstance(p[1], float):
        p[0] = astn.RealValue(p[1])
    else:
        p[0] = astn.IntegerValue(p[1])

def p_operator(p):
    """
    operator : EQ
             | NEQ
             | LT
             | LEQ
             | GT
             | GEQ
    """
    p[0] = p[1]

# errors
def p_error(p):
    print("Syntax error in input!", p)

### YACC END ###

def parse_sql(sql):
    lexer = lex.lex()
    parser = yacc.yacc()
    result = parser.parse(sql, lexer = lexer)
    return result