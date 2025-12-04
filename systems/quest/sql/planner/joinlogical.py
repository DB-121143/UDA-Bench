from quest.sql.nn import *
from quest.core.node import ast_node as astn
from quest.conf import sqlconst
from quest.core.node.logical_node import FilterNode, BinaryNode
import copy
from quest.utils.log import print_log

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def remove_duplicates_columns(lst):
    res = []
    for v in lst:
        flag = False
        for x in res:
            if(x.parse_full() == v.parse_full()):
                flag = True
                break
        if not flag:
            res.append(v)
    return res

class JoinLogicalPlanner(object):
    """
    use a full text search to retrieve document
    """
    def __init__(self):
        self.root = None

    def extract_function(self, attrList):
        res = []
        for v in attrList:
            if isinstance(v, astn.FunctionExpr):
                res.append(copy.copy(v))
        return res

    def extract_binary(self, node):
        res = []
        if not isinstance(node, astn.BinaryOperationExpr):
            raise Exception('Not a Binary!')
        if node.op in sqlconst.LOGIC_TUPLE:
            # AND / OR
            res.extend(self.extract_binary(node.lhs))
            res.extend(self.extract_binary(node.rhs))
        else:
            # >, <, <>, =, >=, <= 
            # lhs is a columnExpr
            res.append(copy.copy(node.lhs))
            if isinstance(node.rhs, astn.ColumnExpr):
                res.append(copy.copy(node.rhs))
        return res

    def extract_attrs_from_whereClause(self, whereClause):
        if isinstance(whereClause, astn.WhereExpr):
            return self.extract_binary(whereClause.value)
        return []

    def extract_binary_condition(self, node):
        res = []
        if not isinstance(node, astn.BinaryOperationExpr):
            raise Exception('Not a Binary!')
        if node.op in sqlconst.LOGIC_TUPLE:
            # AND / OR
            res.extend(self.extract_binary_condition(node.lhs))
            res.extend(self.extract_binary_condition(node.rhs))
        else:
            # >, <, <>, =, >=, <= 
            if isinstance(node.rhs, astn.ColumnExpr):
                now = BinaryNode(node.lhs, node.op, node.rhs)
                res.append(copy.copy(now))
        return res
    
    def extract_conditions_from_whereClause(self, whereClause):
        if isinstance(whereClause, astn.WhereExpr):
            return self.extract_binary_condition(whereClause.value)
        return []
    
    def merge_filter(self, ls : FilterNode, rs : FilterNode):
        # 这里可以直接传引用，因为是左深，合并到一个节点
        ls.add_filter_list(rs.filterList)
        return ls
    
    def build_filter(self, conditions, table):
        if not isinstance(conditions, astn.BinaryOperationExpr):
            raise Exception('Not a Binary!')
        if conditions.op in sqlconst.LOGIC_TUPLE:
            # AND / OR
            ls = self.build_filter(conditions.lhs, table)
            rs = self.build_filter(conditions.rhs, table)
            x = FilterNode(conditions.op, table, [])
            # same AND/OR, can merge
            # diff , append to filterList
            if ls != None:
                if conditions.op == conditions.lhs.op:
                    x = self.merge_filter(x, ls)
                else:
                    x.add_filter(ls)
            if rs != None:
                if conditions.op == conditions.rhs.op:
                    x = self.merge_filter(x, rs)
                else:
                    x.add_filter(rs)
            if len(x.filterList)>0:
                return x
            return None
        else:
            # >, <, <>, =, >=, <= , IN
            if not isinstance(conditions.lhs, astn.ColumnExpr):
                raise Exception('not a ColumnExpr!')
            tbl = conditions.lhs.parse_table()
            now_attr_l = conditions.lhs.parse_column()

            tbr = None
            now_attr_r = None
            # a join cmp
            if isinstance(conditions.rhs, astn.ColumnExpr):
                tbr = conditions.rhs.parse_table()
                now_attr_r = conditions.rhs.parse_column()
                # Not this Table Filter
                if tbl != table and tbr !=table:
                    # dont need to DEFAULT_TABLE_NAME
                    return None

                condi = BinaryNode(conditions.lhs, conditions.op, conditions.rhs)
                return FilterNode('cmp', table, [condi])

            # common filter
            # Not this Table Filter
            if tbl != table and tbl != sqlconst.DEFAULT_TABLE_NAME:
                return None

            condi = BinaryNode(conditions.lhs, conditions.op, conditions.rhs)
            return FilterNode('cmp', table, [condi])
        
    def build_join(self, conditions, joinClause):
        # conditions : list of BinaryNode

        join_type = []
        join_order = []
        m = len(conditions)

        if m>0:
            tableSet = set()
            for v in conditions:
                ltable = v.lhs.parse_table()
                rtable = v.rhs.parse_table()
                tableSet.add(ltable)
                tableSet.add(rtable)
                join_order.append(v)
                join_type.append('INNER')

        if joinClause != None:
            pass
        
        return join_type, join_order

    def build_select(self, selectStmt):
        """
        SELECT [DISTINCT] <select_list>
        FROM <table_source>
        [JOIN <join_condition_list>]
        [WHERE <where_condition>]
        [GROUP BY <group_by_list>]
        [HAVING <having_condition>]
        [ORDER BY <order_by_list>]
        [LIMIT <limit_number>]
        FROM -> JOIN -> WHERE -> GROUP BY -> HAVING -> SELECT -> DISTINCT -> ORDER BY -> LIMIT
        """
        if not isinstance(selectStmt, astn.SelectExpr):
            raise Exception('Not a Select Node!')
        
        # Step0: definition and get retrieve for each table


        where_attrs = self.extract_attrs_from_whereClause(selectStmt.whereClause) # a list of ColumnExpr
        where_attrs = remove_duplicates_columns(where_attrs)
        join_conditions = self.extract_conditions_from_whereClause(selectStmt.whereClause) # join connection with A.t = B.t
        proj_attrs = copy.copy(selectStmt.selectClause.value) # a list of columnExpr and FunctionExpr

        aggr_funcs = self.extract_function(proj_attrs) # aggr functions, a list of FunctionExpr

        all_attrs = []
        all_attrs.extend(where_attrs)
        all_attrs.extend(proj_attrs)
        all_attrs = remove_duplicates_columns(all_attrs)
        tableList = selectStmt.fromClause.value # a list of table name

        # Retrieve
        retrieveDict = {}
        for table in tableList:
            columns = []
            for attr in all_attrs:
                tbl = attr.parse_table()
                if table == tbl or tbl == sqlconst.DEFAULT_TABLE_NAME:
                    columns.append(attr)
            retrieveDict.setdefault(table, LogicalRetrieve(columns = columns, table = table, type = 'Text'))
        
        # Step1: FromClause, get proj_attrs 
        # OrderByClause, get order_attrs GroupByClause, get groupby and having attrs
        
        fromClause = selectStmt.fromClause
        if not isinstance(fromClause, astn.FromExpr):
            raise Exception('No Source From!')
        tableList = fromClause.value

        joinnn = None
        join_type = None
        join_order = None
        if(len(tableList)>1) or selectStmt.joinClause != None:
            """
            TODO
            JoinExpr : 
            JOIN = [JOIN TYPE (INNER/LEFT), TABLE, CONDITION (a binary op)]
            child = [JOIN1, JOIN2, ...]
            """
            join_type, join_order = self.build_join(join_conditions, selectStmt.joinClause)
            joinnn = LogicalJoin(join_type = join_type, join_order = join_order, type = 'Text')
            


        # Step3: whereClase and Extract-Filter

        extractDict = {}
        cnt = 0
        for table in tableList:
            # Filter
            filternn : LogicalFilter
            if len(where_attrs)>0:

                columns = []
                for attr in where_attrs:
                    tbl = attr.parse_table()
                    if table == tbl or tbl == sqlconst.DEFAULT_TABLE_NAME:
                        columns.append(attr)
                filternn = LogicalFilter(columns = columns, table = table, root = self.build_filter(selectStmt.whereClause.value, table))
                    
            # Extract
            columns = []
            extractnn : LogicalExtract
            for attr in proj_attrs:
                tbl = attr.parse_table()
                if table == tbl or tbl == sqlconst.DEFAULT_TABLE_NAME:
                    columns.append(attr)
            extractnn = LogicalExtract(columns = columns, table = table)

            if cnt != 0:
                # transform to full text search
                lhs = join_order[cnt-1].lhs # column expr
                rhs = join_order[cnt-1].rhs
                lcolumn = lhs.parse_full()
                rcolumn = rhs.parse_full()
                ltable = lhs.parse_table()
                rtable = rhs.parse_table()
                if ltable == table:
                    lcolumn, rcolumn = rcolumn, lcolumn # swap, lcolumn should has been extracted

                tmp = LogicalRetrieve(columns = [lcolumn, rcolumn], table = table, type = 'Full')
                tmp.append_input(extractDict[tableList[cnt-1]]) # last table extract
                tmp.append_input(retrieveDict[table])
                print_log("Join Clause: ", ltable, lcolumn, rtable, rcolumn, " ", tableList[cnt-1], " ", extractDict[tableList[cnt-1]])
                if len(where_attrs)>0:
                    filternn.append_input(tmp)
                    extractnn.append_input(filternn)
                else:
                    extractnn.append_input(tmp)
            else:
                if len(where_attrs)>0:
                    filternn.append_input(retrieveDict[table])
                    extractnn.append_input(filternn)
                else:
                    extractnn.append_input(retrieveDict[table])

            extractDict.setdefault(table, extractnn)
            cnt += 1
            
        for node in extractDict.values():
            joinnn.append_input(node)
        

        # Step4 Join Clause, if muilt-table
    

        root = joinnn if joinnn != None else extractDict[tableList[0]]

        # DISTINCT
        

        # ORDER BY
        

        # selectClause Projection

        #projnn = LogicalProjection(proj_attrs)
        #projnn.append_input(root)

        # Limit

        #root = copy.copy(projnn)
        return root

    def build_logical_plan(self, root):
        self.root = self.build_select(root)
        return self.root

