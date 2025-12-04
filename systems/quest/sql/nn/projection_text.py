import pandas as pd
import copy

from .projection import Projection
from quest.utils import column_util, table_util
from quest.core.datapack import *
from quest.conf import sqlconst

class ProjectionText(Projection):
    """
    Info : columns - a list of ColumnExpr
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, type):
        super().__init__(columns, type)
        self.name = 'ProjectionText'
    
    def process(self):
        """
        Porjecion only need to project certain columns
        """
        dataList = []
        for node in self.input:
            dataList.extend(node.get_output())

        # Step 1 : get_datapacks

        full_columns = column_util.parse_column_and_func(self.columns)
        now_table = pd.DataFrame()

        for data in dataList:
            # get table
            if isinstance(data, TablePack):
                now_table = data.table

        for col in now_table.columns:
            if 'doc_id' in col or 'file_name' in col:
                full_columns.append(col)

        if 'count_ALL_COLUMNS_STAR_TABLE.ALL_COLUMNS_STAR' in full_columns:
            full_columns.remove('count_ALL_COLUMNS_STAR_TABLE.ALL_COLUMNS_STAR')
            full_columns.append('Count(*)')
        
        now_table = now_table[full_columns]
        
        if 'doc_id' in now_table.columns:
            # Step 2 : update doc_id to file_name
            index_list = now_table['doc_id'].tolist()
            all_map : dict = self.indexer.get_global_doc_id2file_name()
            #print("total map:\n", all_map)

            now_table = now_table.set_index('doc_id', inplace=False)
            for i in index_list:
                now_table.at[i,'file_name'] = all_map.setdefault(i, 'None')
            #print("index_list:\n",index_list)
        
            now_table.reset_index(inplace=True)
            now_table = now_table.drop(columns='doc_id')

        self.output.append(TablePack('Result', now_table))

        #print("projection_table:\n", now_table)

        return now_table