from .aggregation import Aggregation
from quest.core.datapack import *
from quest.utils import *
import pandas as pd
import copy
from quest.utils.log import print_log
from quest.conf import sqlconst

class AggregationText(Aggregation):
    """
    Info : 
            functions - a list of functionExpr, the apply function
            columns - a list of ColumnExpr, the projection columns
            gp_columns - a list of ColumnExpr, the group by column GROUP BY gp_columns
            type - the extract type ('Photo'/'Text'/..)
    Input : 
    """
    def __init__(self, functions, gp_columns, columns, type):
        super().__init__(functions, gp_columns, columns, type)
        self.name = 'AggregationText'

    def process(self):
        """
        Group by then Aggr, always before projection
        """
        dataList = []
        for node in self.input:
            dataList.extend(node.get_output())

        # Step 1 : get_datapacks

        func_columns = column_util.parse_full(self.functions) # column name
        func = column_util.parse_func_op(self.functions) # column aggr function
        functions = list(zip(func_columns, func))
        gp_columns = column_util.parse_full(self.gp_columns) # group by columns

        #print_log("aggr func_columns : ", func_columns)
        #print_log("aggr func :", func)
        #print_log("aggr gp_columns : ", gp_columns)

        columns = column_util.parse_full(self.columns)
        full_columns = copy.copy(columns)
        full_columns.extend(gp_columns)
        full_columns.append('doc_id')
        full_columns = format_util.remove_duplicates(full_columns)
        now_table = pd.DataFrame(columns=full_columns, index=pd.Index([], name='doc_id'))
        res_doc_list = []

        for data in dataList:
            # get table
            if isinstance(data, TablePack):
                now_table = table_util.merge_table(now_table, data.table, 'doc_id')
            # get text
            if isinstance(data, DocListPack):
                res_doc_list.extend(data.docList)

        res_doc_list = format_util.remove_duplicates(res_doc_list)

        now_table = now_table[full_columns]
        df = now_table[now_table['doc_id'].isin(res_doc_list)].copy()
        if sqlconst.ALL_COLUMNS in func_columns:
            # count(*)
            df[sqlconst.ALL_COLUMNS] = df[sqlconst.ALL_COLUMNS].fillna(1)
        print_log("now_table to aggr: ",df)

        return df
        
        # Step 2 : aggr
        now_table = table_util.aggregation_table_transform(df, functions, gp_columns) # always only one gp_column
        self.output.append(TablePack('Result', now_table))

        print("aggr_table:\n", now_table)

        return now_table
        