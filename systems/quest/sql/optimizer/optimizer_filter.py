from quest.sql.nn import *
from quest.utils import *
from quest.core.node import ast_node as astn
from quest.core.node.logical_node import FilterNode, BinaryNode
from quest.conf import settings, sqlconst
import copy
from quest.utils.log import print_log
import functools
import pandas as pd
import numpy as np
from quest.utils.class2json import ClassToJson
import random

# 1 = seletivity 2 = cost
    # (1-seletivity)/cost with descending order for AND
    # (seletivity)/cost with desc for OR
    # list[1] = sele/cost


# QUEST
def cmp_and(x, y):
    # default sort from small to big, we need (1-sele)/cost from big to small
    tx = (1-x[1])/x[2] 
    ty = (1-y[1])/y[2]
    if abs(tx - ty) < 1e-8:
        return 0
    if tx > ty:
        return 1
    return -1 

def cmp_or(x, y):
    # default sort from small to big, we need sele/cost from big to small
    tx = x[1]/x[2] 
    ty = y[1]/y[2]
    if abs(tx - ty) < 1e-8:
        return 0
    if tx > ty:
        return 1
    return -1 

"""
# PZ
def cmp_and(x, y):
    tx = x[1]
    ty = y[1]
    if abs(tx-ty) < 1e-8:
        return 0
    if tx < ty:
        return 1
    return -1

def cmp_or(x, y):
    tx = x[1]
    ty = y[1]
    if abs(tx-ty) < 1e-8:
        return 0
    if tx > ty:
        return 1
    return -1
"""


"""
# zenDB
def cmp_and(x, y):
    tx = x[1] * x[2]
    ty = y[1] * y[2]
    if abs(tx-ty) < 1e-8:
        return 0
    if tx < ty:
        return 1
    return -1

def cmp_or(x, y):
    tx = x[1] * x[2]
    ty = y[1] * y[2]
    if abs(tx-ty) < 1e-8:
        return 0
    if tx < ty:
        return 1
    return -1
"""


def cmp_join(x, y):
    # selectivity smaller is better
    tx = x[1]
    ty = y[1]
    if abs(tx - ty) < 1e-8:
        return 0
    if tx > ty:
        return 1
    return -1 

class OptimizerFilter(object):
    """
    OptimizerFilter need to rebuild the filter and the retrieve node
    """

    def __init__(self, indexer, sampler, querier, batch_size=30):
        self.root = None
        self.global_indexer = indexer # to attach nn with document indexer
        self.sampler = sampler
        self.querier = querier
        self.batch_size = batch_size
        self.now_retrieve_docs = []
        self.visited = []
        # optimize nodes!

    def clear(self):
        for v in self.visited:
            v.visited = False
        self.visited = []

    def calc(self, filter : BinaryNode, table):
        """
        filter : a BinaryNode
        TODO 这里主要还需要考虑多表的情况，先放着
        return : seletivity, cost
        """
        #return 1,10

        cst = 0
        sel = 0
        df = copy.copy(self.sampler.sample_table)

        ltable = filter.lhs.parse_table()
        col = filter.lhs.parse_full()
        if ltable != table and ltable != sqlconst.DEFAULT_TABLE_NAME:
            col = filter.rhs.parse_full()
        
        indexer, typ = self.global_indexer.get_indexer(table)
        print(ltable, " ", table, " ---col--- ", col," filter: ", filter.parse())
        cst = indexer.get_relative_chunks_lenght(self.now_retrieve_docs, col, topk = settings.TOPK)

        # Here may need to use a better check, but it is always useful!
        if isinstance(filter.rhs, astn.ColumnExpr):
            """
            不妨认为表之前的比较，选择性很低
            """
            return random.uniform(0, 0.01), cst
        
        if isinstance(filter.rhs, astn.StringValue) and (filter.op in settings.VALUE_OP):
            return random.uniform(0, 0.05), cst

        # 字符串相等或者数字比较可以用query

        lhs = filter.lhs.parse_full()
        if isinstance(filter.rhs, astn.IntegerValue):
            #print("downcast: integer")
            df[lhs] = pd.to_numeric(df[lhs], errors='coerce')
            df.dropna(subset=[lhs], inplace=True)
            df[lhs] = df[lhs].astype(int)
            #print("after adjust type lhs:\n", df[lhs])
        elif isinstance(filter.rhs, astn.RealValue):
            #print("downcast: float")
            df[lhs] = pd.to_numeric(df[lhs], errors='coerce')
            df.dropna(subset=[lhs], inplace=True)
            df[lhs] = df[lhs].astype(float)
            #print("after adjust type lhs:\n", df[lhs])

        print("try calc-- sample table:\n", df)
        #print(df.dtypes)
        condition = '`' + filter.lhs.parse_full() + '`' + ' ' + filter.op + ' ' + str(filter.rhs.parse_full())
        ndf = df.query(condition) # here should be in sampler

        if df.shape[0] == 0:
            sel = 1.0
        else:
            sel = float(ndf.shape[0]) / float(df.shape[0]) + 0.05

        #print("new df:\n",ndf)
        #print("sel:{}, cst:{}".format(sel, cst))
        #print(ndf.shape[0], " ", df.shape[0])
        
        return sel, cst
    
    def optimze_filter_tree(self, node, table):
        """
        optimize the filter order of all doc_id in self.now_retrieve_docs
        node : filter tree
        table : now table
        """
        res = copy.copy(node)
        if not isinstance(node, FilterNode):
            raise Exception('Not a FilterNode!')
        if node.type == 'cmp':
            """
            这里需要注意处理多表rhs是col的情况，我们先忽略选择性
            rhs是col一定会在最后一个
            """
            sel, cst = self.calc(node.filterList[0], table)
            return res, sel, cst
        else:
            son = []
            for v in node.filterList:
                nod, sel, cst = self.optimze_filter_tree(v, table) 
                son.append([nod, sel, cst])
            
            now_cost = 0
            now_seletivity = 1
            now_prod = 1
            # sort desc
            if node.type == 'AND':
                #print("before:", son)
                son.sort(key=functools.cmp_to_key(cmp_and), reverse=True)
                #print("after:", son)
                for v in son:
                    now_cost = now_cost + v[2] * now_prod
                    now_prod = now_prod * v[1]
                now_seletivity = now_prod 
            else:
                son.sort(key=functools.cmp_to_key(cmp_or), reverse=True)
                for v in son:
                    now_cost = now_cost + v[2] * now_prod
                    now_prod = now_prod * (1 - v[1])
                now_seletivity = 1 - now_prod 
            
            filters = []
            for v in son:
                filters.append(v[0])

            #print("filters : ", filters)

            res.set_filterList(filters)

            return res, now_seletivity, now_cost

    def build_retrieve(self, root : LogicalRetrieve):
        # default retrieve build as all

        # embedding node
        if root.type != 'Text':
            node = copy.copy(root)
            node.visited = False
            return [node]

        # Step 1 : Split retrieve by batch!
        tmpIndexer, typ = self.global_indexer.get_indexer(root.table)

        if tmpIndexer is None:
            raise Exception('No Such Indexer for table: ' + root.table)
        
        allList = tmpIndexer.get_docs_id()
        splitList = format_util.batch_split(allList, self.batch_size)

        # Step 2 : Build retrieve node for each batch

        res = []
        for batch in splitList:
            node = RetrieveText(root.columns, root.table, root.type)
            node.set_indexer(tmpIndexer)
            node.set_retrieveList(batch)
            node.set_sampler(self.sampler)
            res.append(node)
        return res

    def build_filter(self, root : LogicalFilter):
        # trt optimize filter 

        son = root.input[0]
        if not isinstance(son, LogicalRetrieve):
            print_log("OptimizerFilter: Cannot optimize filter for non-retrieve node, return original node!")
            return root

        # if son is retrieve, then we can optimize
        # Step 1 : Split retrieve by batch!

        retrieve_node_list = self.build_retrieve(son)

        # Step 2 : build filter node and optimize the filter tree

        filter_node_list = []
        for retrieve_node in retrieve_node_list:
            if retrieve_node.type != 'Text':
                continue
            # optimize
            self.now_retrieve_docs = retrieve_node.retrieveList
            newroot, now_seletivity, now_cost= self.optimze_filter_tree(copy.copy(root.root), root.table)

            jsonConverter = ClassToJson()
            js = jsonConverter.toJson(newroot)
            #print_log("Filter Tree:\n",js)

            # TODO : check if we need a semantics filter
            node = FilterText(root.columns, root.table, 'Text', newroot)
            node.set_querier(self.querier)
            node.append_input(retrieve_node)  
            filter_node_list.append(node)

        return filter_node_list
    
    def build_extract(self, root : LogicalExtract):
        node = copy.copy(root)
        node.visited = False
        return node
    
    def build_projection(self, root : LogicalProjection):
        node = copy.copy(root)
        node.visited = False
        return node
    
    def build_aggregation(self, root : LogicalAggregation):
        node = copy.copy(root)
        node.visited = False
        return node
    
    def build_group(self, root : LogicalGroup):
        node = copy.copy(root)
        node.visited = False
        return node

    def build_join(self, root : LogicalJoin):
        node = copy.copy(root)
        node.visited = False
        return node

    def dfs_build(self, root):
        if root.visited or isinstance(root, Physical):
            return root
        root.visited = True
        self.visited.append(root)
        
        now_input = []
        for x in root.input:
            son = self.dfs_build(x)
            if isinstance(son, list):
                now_input.extend(son)
            else:
                now_input.append(son)

        # build root
        res = None
        for name in sqlconst.nnType:
            if not name in root.__class__.__name__.lower():
                continue
            func_name = 'build_' + name
            #print("now try process build: ", func_name)
            method = getattr(self, func_name, None)
            if callable(method):
                #print("now process build: ", func_name)
                res = method(root)
            else:
                raise Exception('No method found for class {root.__class__.__name__}')
        if not isinstance(res, list):
            res.set_input(now_input)
        return res
    
    def build(self, root):
        self.root = self.dfs_build(root)
        self.clear()
        return self.root