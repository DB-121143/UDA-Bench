from quest.sql.nn import *
from quest.core.node import ast_node as astn
from quest.conf import sqlconst
import copy
from quest.utils.log import print_log


class JoinPhysicalPlanner(object):
    """
    convert the rest logical to physical
    this is the true version for complex plan
    change the retireve type to Text/Embedding(Doc/Chunk)/Full
    """
    def __init__(self, indexer, querier, sampler):
        self.root = None
        self.global_indexer = indexer# to attach nn with document indexer
        self.querier = querier
        self.sampler = sampler
        self.order = []
        self.res = []

    def build_retrieve(self, root : LogicalRetrieve):
        # default retrieve build as all
        node = None
        if root.type == 'Text':
            node = RetrieveText(root.columns, root.table, 'Text')
        elif root.type =='Embedding':
            node = RetrieveEmbedding(root.columns, root.table, 'Doc')
        elif root.type == 'Full':
            node = RetrieveFull(root.columns, root.table, 'Full')

        ind, typ = self.global_indexer.get_indexer(root.table)
        node.set_indexer(ind)
        node.set_retrieveList(node.indexer.get_docs_id())
        node.set_sampler(self.sampler)

        return node

    def build_filter(self, root : LogicalFilter):
        node = FilterText(root.columns, root.table, 'Text', root.root)
        node.set_querier(self.querier)
        return node
    
    def build_extract(self, root : LogicalExtract):
        node = ExtractText(root.columns, root.table, 'Text')
        node.set_querier(self.querier)
        return node
    
    def build_projection(self, root : LogicalProjection):
        node = ProjectionText(root.columns, 'Text')
        node.set_indexer(self.global_indexer)
        return node
    
    def build_join(self, root : LogicalJoin):
        node = JoinText(root.join_type, root.join_order, 'Text')
        node.set_indexer(self.global_indexer)
        return node
    
    def build_aggregation(self, root : LogicalAggregation):
        node = AggregationText(root.functions, root.gp_columns, root.columns, 'Text')
        return node
    
    def build_group(self, root: LogicalGroup):
        node = GroupText(root.columns, root.table, root.N_clusters, 'Text')
        node.set_querier(self.querier)
        return node
    
    def dfs_order(self, root):
        print_log("visit node: ", root, "with size: ", len(root.input))
        root.visited = 1
        for x in root.input:
            if x.visited == 0:
                self.dfs_order(x)
        root.visited = 2
        self.order.append(root)
    
    def dfs_build(self, root):
        if root.visited or isinstance(root, Physical):
            return root
        root.visited = True
        
        now_input = []
        for x in root.input:
            son = self.dfs_build(x)
            if isinstance(son, list):
                now_input.extend(son)
            else:
                now_input.append(son)

        # build root to Physical
        res = None
        for name in sqlconst.nnType:
            if not name in root.__class__.__name__.lower():
                continue
            func_name = 'build_' + name
            print("now try process build: ", func_name)
            method = getattr(self, func_name, None)
            if callable(method):
                print("now process build: ", func_name)
                res = method(root)
            else:
                raise Exception('No method found for class {root.__class__.__name__}')
        
        res.set_input(now_input)
        return res
    
    def build(self, root):
        self.dfs_order(root)
        self.res = [0] * len(self.order)
        for i, v in enumerate(self.order):
            result = None
            v.visited = i
            for name in sqlconst.nnType:
                if not name in v.__class__.__name__.lower():
                    continue
                func_name = 'build_' + name
                print("now try process build: ", func_name)
                method = getattr(self, func_name, None)
                if callable(method):
                    print("now process build: ", func_name)
                    result = method(v)
                else:
                    raise Exception('No method found for class {v.__class__.__name__}')
                
            now_input = []
            for x in v.input:
                son = self.res[x.visited]
                if isinstance(son, list):
                    now_input.extend(son)
                else:
                    now_input.append(son)
            result.set_input(now_input)
            self.res[i] = result
            
        self.root = self.res[-1]
        return self.root




