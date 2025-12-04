from quest.sql.nn import *
from quest.utils import *
from quest.core.node import ast_node as astn
from quest.core.node.logical_node import FilterNode, BinaryNode
from quest.conf import sqlconst
import copy
from quest.utils.log import print_log
import functools
import pandas as pd
import numpy as np

def cmp_join_filternum(x, y):
    # filternum bigger is better
    tx = x[1]
    ty = y[1]
    if abs(tx - ty) < 1e-8:
        return 0
    if tx > ty:
        return 1
    return -1 

def cmp_join(x, y):
    # doc/filter_number smaller is better
    tx = x[1]
    ty = y[1]
    if abs(tx - ty) < 1e-8:
        return 0
    if tx < ty:
        return 1
    return -1 

class OptimizerJoin(object):
    """
    OptimizerFilter need to rebuild the filter node order
    simple use the length of filter to select first table
    more filter means less rows
    or use docs/filter_number smaller is OK
    """

    def __init__(self, indexer, sampler, querier, batch_size=100):
        self.root = None
        self.global_indexer = indexer # to attach nn with document indexer
        self.sampler = sampler
        self.querier = querier
        self.batch_size = batch_size
        self.now_retrieve_docs = []
        self.visited = []
        self.filter_nodes = []
        # optimize nodes!

    def clear(self):
        for v in self.visited:
            v.visited = False
        self.visited = []

    def build_retrieve(self, root : LogicalRetrieve):
        node = copy.copy(root)
        node.visited = False
        return node

    def build_filter(self, root : LogicalFilter):
        # trt optimize filter 
        node = copy.copy(root)
        node.visited = False
        self.filter_nodes.append(node)
        return node  
    
    def build_extract(self, root : LogicalExtract):
        node = copy.copy(root)
        node.visited = False
        return node
    
    def build_projection(self, root : LogicalProjection):
        node = copy.copy(root)
        node.visited = False
        return node

    def build_join(self, root : LogicalJoin):
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
        print(root)
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

        # connect filters

        for i, node in enumerate(self.filter_nodes):
            if i==0:
                continue
            # connect join nodes[i-1] to nodes[i]
            self.filter_nodes[i].append_input(self.filter_nodes[i-1])

        return self.root