from .filter import Filter
from quest.core.datapack import *
from quest.utils import *
import pandas as pd
import copy
from quest.core.llm.llm_query import TextLLMQuerier
from quest.core.node.logical_node import BinaryNode, FilterNode
from quest.core.node import ast_node as astn
from quest.utils.log import print_log

def list_and(x, y):
    if x == None:
        x = []
    if y == None:
        y = []
    return list(set(x) & set(y))

def list_or(x, y):
        if x == None:
            x = []
        if y == None:
            y = []
        return list(set(x) | set(y))

def list_xor(x, y):
    if x == None:
        x = []
    if y == None:
        y = []
    return list(set(x) ^ set(y))

class FilterText(Filter):
    """
    Info : columns - a list of ColumnExpr
            table - the filter tableName
            type - the filter type ('Photo'/'Text'/..)
            root - filter tree root
    Input : None
    """
    def __init__(self, columns, table, type, root):
        super().__init__(columns, table, type, root)
        self.name = 'FilterText'
        self.textDict = {} # {doc_id1 : { column1 :[text1, text2, ...], }
        self.now_tableDict = {}

    def solve(self, node : FilterNode, doc_idList):
        """
        dfs the filter tree, and get the result table 
        node : now tree node
        doc_idList : the doc_id list to be filter

        update --
        self.now_table : the result table after filter, may include muiltiple dataframe ()

        return --
        res_doc_idList : the rest doc_id list after filter list[int]


        You should use now_doc_idList to check filed
        use res_doc_idList to return
        """
        res_doc_idList = []
        now_doc_idList = copy.copy(doc_idList)
        if node.type == 'AND':
            for v in node.filterList:
                res_doc_idList = self.solve(v, now_doc_idList)
                now_doc_idList = copy.copy(res_doc_idList)
            return res_doc_idList
        elif node.type == 'OR':
            for v in node.filterList:
                res_doc_idList = list_or(res_doc_idList, self.solve(v, now_doc_idList))
                now_doc_idList = list_xor(res_doc_idList, doc_idList)
            return res_doc_idList
        else:
            # cmp try filter
            filter : BinaryNode = copy.copy(node.filterList[0])
            condition = filter.parse()
            condition = '`' + filter.lhs.parse_full() + '`' + ' ' + filter.op + ' ' + str(filter.rhs.parse_full())
            lcolumn = filter.lhs.parse_full()
            ltable = filter.lhs.parse_table()
            now_column = lcolumn # the filter column
            now_table = ltable

            print_log("\n now try filter - ", condition, " \n with space - ", now_doc_idList, "\n")
            
            # Extract first
            if filter.op != 'IN' and not isinstance(filter.rhs, astn.ColumnExpr) and not isinstance(filter.rhs, astn.StringValue):
                # if a normal value filters
                
                # Step 0 : get now textDict

                nowDict = table_util.check_dict_and_table(self.textDict, now_doc_idList, [lcolumn], self.now_tableDict[self.table])
                #nowDict = check_dict(self.textDict, now_doc_idList)

                # Step 1 : extract data
                df = self.querier.extract_attribute_from_textDict(nowDict, [lcolumn])

                # Step 2 : format the column
                if isinstance(filter.rhs, astn.IntegerValue):
                    #print("downcast: integer")
                    df[lcolumn] = pd.to_numeric(df[lcolumn], errors='coerce')
                    df.dropna(subset=[lcolumn], inplace=True)
                    df[lcolumn] = df[lcolumn].astype(int)
                elif isinstance(filter.rhs, astn.RealValue):
                    #print("downcast Fload")
                    df[lcolumn] = pd.to_numeric(df[lcolumn], errors='coerce')
                    df.dropna(subset=[lcolumn], inplace=True)
                    df[lcolumn] = df[lcolumn].astype(float)

                #print("after adjust type lhs:\n", df)

                # Step 3 : merge
                self.now_tableDict[self.table] = table_util.merge_table(self.now_tableDict[self.table], df, key='doc_id')
                #print("after merge:\n", self.now_tableDict[self.table])

                # Step 4 : check filter now
                now_data_table  = copy.copy(self.now_tableDict[self.table])
                now_data_table = now_data_table.set_index('doc_id', inplace = False)
                exist_idx = [idx for idx in now_doc_idList if idx in now_data_table.index] # check exist
                now_data_table = now_data_table.loc[exist_idx]
                now_data_table.reset_index(inplace=True)
                now_data_table = now_data_table.query(condition)
                
                #print("after filter:\n", now_data_table)

                # Step 5 : check the filed include
                res_doc_idList = format_util.remove_duplicates(now_data_table['doc_id'].tolist())
                res_doc_idList = list(map(int,res_doc_idList))
                #print("res_doc_id_List: ", res_doc_idList)
            
            else:
                # if columnExpr or IN need semantics

                if isinstance(filter.rhs, astn.ColumnExpr):
                    # we dont apply in here
                    # need to extract now_column in 
                    rcolumn = filter.rhs.parse_full()
                    rtable = filter.rhs.parse_table()
                    last_table = rtable
                    last_column = rcolumn
                    if self.table != ltable:
                        now_column = rcolumn
                        now_table = rtable
                        last_table = ltable
                        last_column = lcolumn

                    # if last_table not visited, only need to extract here
                    if last_table not in self.now_tableDict:
                        nowDict = table_util.check_dict_and_table(self.textDict, now_doc_idList, [now_column], self.now_tableDict[now_table])
                        #nowDict = check_dict(self.textDict, now_doc_idList)
                        df = self.querier.extract_attribute_from_textDict(nowDict, [now_column])
                        self.now_tableDict[self.table] = table_util.merge_table(self.now_tableDict[self.table], df, key='doc_id')
                        self.now_tableDict[self.table] = table_util.fill_cells(self.now_tableDict[self.table], [now_column], now_doc_idList, 'doc_id')
                        
                        return now_doc_idList
                    
                    # else process IN filter

                    # get already in value
                    df = copy.copy(self.now_tableDict[last_table])
                    df_list = df[last_column].tolist()
                    condition = str(now_column) + 'IN [' + ', '.join(df_list) + ']'

                    #print("condition : ", condition, '---- column : ',now_column)

                    nowDict = table_util.check_dict_and_table(self.textDict, now_doc_idList, [now_column], self.now_tableDict[now_table])
                    #nowDict = check_dict(self.textDict, now_doc_idList)
                    df = self.querier.extract_attribute_from_textDict_semantic_fiter(nowDict, [now_column], condition)

                else:
                    # filter IN, or filter '=='string''
                    # rhs is astn.ListValue or a StringValue
                    lst = filter.rhs.parse_full()
                    condition = None
                    if isinstance(lst, list):
                        condition = str(now_column) + 'IN [' + ', '.join(lst) + ']'
                    elif isinstance(lst, str):
                        condition = str(now_column) + filter.op + lst 
                    else:
                        condition = lst

                    print("condition : ", condition, '---- column : ',now_column)

                    nowDict = table_util.check_dict_and_table(self.textDict, now_doc_idList, [now_column], self.now_tableDict[self.table])
                    #nowDict = check_dict(self.textDict, now_doc_idList)
                    df = self.querier.extract_attribute_from_textDict_semantic_fiter(nowDict, [now_column], condition)

                # drop fcondition

                #print_log("get filter df : \n", df)

                ndf = df.drop(columns = 'fcondition')
                #print_log("ndf : \n", ndf)

                # Step 3 : merge
                #print_log("before merge tableDict : \n", self.now_tableDict[self.table])

                self.now_tableDict[self.table] = table_util.merge_table(self.now_tableDict[self.table], ndf, key='doc_id')

                #print_log("after merge tableDict : \n", self.now_tableDict[self.table])

                # Step 4 : check filter

                # Step 4-1 get the exist doc_ids that extract before
                ndf = df.set_index('doc_id', inplace = False)
                #print_log("after set index ndf : \n", ndf)
                checked_idx = [idx for idx in now_doc_idList if idx in ndf.index] # only filter this time docs index
                exist_idx = list_xor(now_doc_idList, checked_idx) # extract before, need to check samantics
                ndf.reset_index(inplace = True) # remember to reset
                
                #print_log("now_doc_idList : --- \n", now_doc_idList)
                #rint_log("checked_idx : ",checked_idx)
                #print_log("exist_idx : ", exist_idx)

                # Step 4-2 get the exist value
                gb_table = copy.copy(self.now_tableDict[self.table])
                gb_table = gb_table.set_index('doc_id', inplace = False)
                exist_check_text = gb_table.loc[exist_idx, now_column].tolist()
                exist_df = self.querier.check_filter_condition(exist_check_text, exist_idx, [now_column], condition)
                gb_table.reset_index(inplace=True) # remember to reset

                #print_log("exist_df : \n", exist_df)

                df = table_util.merge_table(df, exist_df, key='doc_id') # filter this time merge fiter before with fcondition!

                #print_log("merge df and exist_df : \n", df)

                df = df[df['fcondition'].apply(lambda x: str(x).strip() == 'True')]
                #df = df[df['fcondition'].isin(['True', True])]

                #print_log("after check fcondition df : \n", df)

                # Step 5 : check the filed include
                if df.empty:
                    res_doc_idList = []
                else:
                    res_doc_idList = format_util.remove_duplicates(df['doc_id'].tolist())
                    res_doc_idList = list(map(int,res_doc_idList))
                #print_log("res_doc_idList : ",res_doc_idList)

            # fill cells , not extract again!
            self.now_tableDict[self.table] = table_util.fill_cells(self.now_tableDict[self.table], [now_column], now_doc_idList, 'doc_id')

            return res_doc_idList

    def process(self):
        """
        Filter need to get: docList (from Retrieve or Filter), table (from Filter)
        """
        dataList = []
        for node in self.input:
            dataList.extend(node.get_output())

        # step 1 : get_datapacks

        columns = column_util.parse_full(self.columns)
        full_columns = copy.copy(columns)
        full_columns.append('doc_id')
        self.now_tableDict[self.table] = pd.DataFrame(columns=full_columns, index=pd.Index([], name='doc_id'))
        #print_log("before data pack", self.now_tableDict[self.table])

        for data in dataList:

            # get table, note that may include other tables
            if isinstance(data, TablePack):
                now_table = data.tablename 
                if now_table != self.table:
                    # may check if exist
                    if now_table in self.now_tableDict.keys():
                        self.now_tableDict[now_table] = table_util.merge_table(self.now_tableDict[now_table], data.table)
                    else:
                        self.now_tableDict[now_table] = copy.copy(data.table)
                else:
                    self.now_tableDict[now_table] = table_util.merge_table(self.now_tableDict[now_table], data.table)

            # get text
            if isinstance(data, TextPack):
                doc_id = data.doc_id
                column = data.column
                text = data.text
                self.textDict.setdefault(doc_id, {})
                self.textDict[doc_id].setdefault(column, [])
                self.textDict[doc_id][column].append(text)
            
            if isinstance(data, TextListPack):
                doc_id = data.doc_id
                column = data.column
                text = data.textList
                self.textDict.setdefault(doc_id, {})
                self.textDict[doc_id].setdefault(column, [])
                self.textDict[doc_id][column].extend(text)

            if isinstance(data, TextDictPack):
                doc_id = data.doc_id
                text = data.textDict
                self.textDict.setdefault(doc_id, text)

        #print_log("after data pack", self.now_tableDict[self.table])

        doc_idList = list(self.textDict.keys())
            
        # step2 : filter from text

        # step2-1 : access local database and get cache

        # step2-2 : process filter tree
        res_doc_idList = self.solve(self.root, doc_idList)
        
        # step3 : output
        self.output.append(TablePack(self.table, self.now_tableDict[self.table]))

        #print_log("after filter res docs---: \n", res_doc_idList)

        # {doc_id1 : { column1 :[text1, text2, ...], }
        for tid, value in self.textDict.items():
            if tid not in res_doc_idList:
                continue

            #self.output.append(TextDictPack(tid, value))
            #print_log("continue to extract doc id is : ", tid)
            for col, lst in value.items():
                #print_log("append - ", tid, ' ' , lst, ' ', col, ' --- ', columns)
                self.output.append(TextListPack(tid, lst, col))

        #print("filter_table:\n", self.now_tableDict[self.table])
        