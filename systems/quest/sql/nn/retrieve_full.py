from .retrieve import Retrieve
from quest.utils import *
from quest.core.datapack import *
import quest.conf.settings as settings
from quest.utils.log import print_log
import copy
import pandas as pd

class RetrieveFull(Retrieve):
    """
    Info : columns - a list of ColumnExpr
            table - the retrieve tableName
            type - the retrieve type ('Doc'/'Chunk'/..)

            self.retreieveList
            self.indexer
            self.sampler
    Input : None
    """
    def __init__(self, columns, table, type):
        super().__init__(columns, table, type)
        self.name = 'RetrieveFull'

    def set_querier(self, x):
        self.querier = x

    def process(self):
        """
        full text retrieve by a table and a columns[0] has been extracted

        """

        dataList = []
        for node in self.input:
            print_log("retrieve full has a child : ", node)
            dataList.extend(node.get_output())

        # step 1 : get_datapacks
        lcolumn = self.columns[0]
        rcolumn = self.columns[1]
        textDict = {} # {doc_id1 : { column1 :[text1, text2, ...], }
        now_table = pd.DataFrame()

        print_log("retrieve full input dataList : ", dataList)

        for data in dataList:

            # get table
            if isinstance(data, TablePack):
                now_table =  data.table

            # get text
            if isinstance(data, TextPack):
                doc_id = data.doc_id
                column = data.column
                text = data.text
                textDict.setdefault(doc_id, {})
                textDict[doc_id].setdefault(column, [])
                textDict[doc_id][column].append(text)
            
            if isinstance(data, TextListPack):
                doc_id = data.doc_id
                column = data.column
                text = data.textList
                textDict.setdefault(doc_id, {})
                textDict[doc_id].setdefault(column, [])
                textDict[doc_id][column].extend(text)

            if isinstance(data, TextDictPack):
                doc_id = data.doc_id
                text = data.textDict
                textDict.setdefault(doc_id, text)


        # get value in now_table
        print_log("retrieve full now_table : \n", now_table)
        values = now_table[lcolumn].drop_duplicates().tolist()

        print_log("values : \n", values)

        res_tuple = self.indexer.full_text_search_related_docs(queries  = values, threshold = settings.RETRIEVE_FULL_THRESHOLD) # doc_id[0] is true

        print_log("retrieve tuple: \n", res_tuple)

        res_doc_idList = []
        for tup_list in res_tuple:
            for tup in tup_list:
                res_doc_idList.append(tup[0]) # tup[0] is doc_id

        res_doc_idList = list(set(res_doc_idList))

        # {doc_id1 : { column1 :[text1, text2, ...], }
     
        # Step 2 : Get text List and Pack

        for tid, value in textDict.items():
            if tid not in res_doc_idList:
                continue

            #self.output.append(TextDictPack(tid, value))
            #print_log("continue to extract doc id is : ", tid)
            for col, lst in value.items():
                #print_log("append - ", tid, ' ' , lst, ' ', col, ' --- ', columns)
                self.output.append(TextListPack(tid, lst, col))

                



        

