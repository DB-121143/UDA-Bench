from .base import Base
from .extract import Extract
from quest.core.datapack import *
from quest.utils import *
import pandas as pd
import copy
from quest.core.llm.llm_query import TextLLMQuerier
from quest.utils.log import print_log

class ExtractText(Extract):
    """
    Info : columns - a list of ColumnExpr
            table - the extract tableName
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, table, type):
        super().__init__(columns, table, type)
        self.name = 'ExtractText'

    def process(self):
        """
        Extract need to get: docList (from Retrieve or Filter), table (from Filter)
        """
        dataList = []
        for node in self.input:
            dataList.extend(node.get_output())

        # step 1 : get_datapacks
        columns = column_util.parse_full(self.columns)
        full_columns = copy.copy(columns)
        full_columns.append('doc_id')
        textDict = {} # {doc_id1 : { column1 :[text1, text2, ...], }
        now_table = pd.DataFrame(columns=full_columns, index=pd.Index([], name='doc_id'))

        for data in dataList:

            # get table
            if isinstance(data, TablePack):
                now_table = table_util.merge_table(now_table, data.table, 'doc_id')

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
        
        #print_log("extract after data pack : \n", now_table)

        #print_log("accessed text dict : \n", textDict)

        res_doc_list = list(textDict.keys())
        res_doc_id_list = [int(x) for x in res_doc_list]

        #rint_log("res_doc_list:", res_doc_list)
            
        # step2 : extract from text

        # step2-1 : access local database and get cache
        new_textDict = table_util.check_dict_and_table(textDict, res_doc_id_list, columns, now_table)

        #print_log("need to extract text dict : \n", new_textDict)

        # delete no used ones

        # step2-2 : query input, build the input and the query in LLM
        df = self.querier.extract_attribute_from_textDict(textDict = new_textDict, attributeList = columns)

        # step2-3 : merge

        now_table = table_util.merge_table(now_table, df, key='doc_id')
        #print_log("merge extract_table:\n", now_table)

        # step3 : output

        now_table = table_util.keep_table(now_table, res_doc_id_list, 'doc_id')

        print_log("fianl_table:\n",now_table)

        self.output.append(TablePack(self.table, now_table))
        self.output.append(DocListPack(self.table, res_doc_id_list))

        return None
        