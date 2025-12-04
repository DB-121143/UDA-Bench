from .base import Base
from .group import Group
from quest.core.datapack import *
from quest.utils import *
import pandas as pd
import copy
from quest.core.llm.llm_query import TextLLMQuerier
from quest.utils.log import print_log
from quest.core.nlp import text_cluster
from quest.conf import settings
import random
import numpy as np
from collections import Counter

def reverse_dict(d : dict):
    res = {}
    for key, value in d.items():
        res.setdefault(value, [])
        res[value].append(key)
    return res

def check_most(vec : list):
    counts = Counter(vec)
    return counts.most_common(1)[0][0]

class GroupText(Group):
    """
    Info : columns - a list of ColumnExpr, the group by columns
            table - the extract tableName
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, columns, table, N_clusters, type):
        super().__init__(columns, table, N_clusters, type)
        self.name = 'GroupText'
        self.querier = None

    def process(self):
        """
        Group need to get: docList (from Retrieve or Filter)
        should return textPacks
        """
        dataList = []
        for node in self.input:
            dataList.extend(node.get_output())

        # step 1 : get_datapacks
        columns = column_util.parse_full(self.columns)
        full_columns = copy.copy(columns) # always only group by column and doc_id
        full_columns.append('doc_id')
        textDict = {} # {doc_id1 : { column1 :[text1, text2, ...], }
        embeddingDict = {} # {doc_id1 : emb1, .. }
        now_table = pd.DataFrame(columns=full_columns, index=pd.Index([], name='doc_id'))

        res_doc_list = []
        for data in dataList:
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

            if isinstance(data, EmbeddingPack):
                doc_id = data.doc_id
                embedding = data.embedding
                column = data.column # always no used
                embeddingDict.setdefault(doc_id, embedding)
            
            if isinstance(data, DocListPack):
                res_doc_list = data.docList

        doc_idList = []
        if res_doc_list:
            doc_idList = res_doc_list
        else:
            doc_idList = list(textDict.keys())

        for data in dataList:
            if isinstance(data, EmbeddingListPack):
                doc_id = data.doc_id
                if doc_id not in doc_idList:
                    continue
                embeddingList = data.embeddingList
                embedding = math_util.calc_center(embeddingList)
                column = data.column # always no used
                embeddingDict.setdefault(doc_id, embedding)

        #print_log("embeddingDict : ", embeddingDict)
        #print_log("doc_idList : ", doc_idList)

        # Step 2 group by embedding

        doc_to_group = text_cluster.cluster_docs_by_embedding(embeddingDict, self.N_clusters, 998244353) # {doc_id : group}

        #print_log("doc_to_group after cluster : ", doc_to_group)

        group_to_docList = reverse_dict(doc_to_group) 

        print_log("group_to_docList : ",group_to_docList)

        # Step 3 sample are extract

        res_idList = []
        sample_dict = {}
        group_value = {} # group -> value
        for key, docList in group_to_docList.items():
            # sample for group key
            N = len(docList)
            sample_num = min(N, settings.GROUP_SAMPLE_NUM)
            sampler_ids = random.sample(docList, sample_num)
            res_idList.extend(sampler_ids)
            sample_dict.setdefault(key, sampler_ids)

        new_textDict = table_util.check_dict_and_table(textDict, res_idList, columns, now_table)

        df = self.querier.extract_attribute_from_textDict(textDict = new_textDict, attributeList = columns)

        print_log("group extracted value df : \n",df)

        # Step 3-1 get the real value for each group

        for key, docList in sample_dict.items():
            filtered_values = df.loc[df['doc_id'].isin(docList), columns[0]]
            final_value = check_most(filtered_values)
            group_value.setdefault(key, final_value)

       # print_log("group most value dict : \n", group_value)

        # Step 3-2 apply value for each group

        now_columns = df.columns

        new_rows = pd.DataFrame(
            {col: [np.nan] * len(doc_idList) if col != 'doc_id' else doc_idList for col in now_columns}
        )

        now_table = pd.concat([now_table, new_rows], ignore_index=True)
        print_log(now_table)

        for key, docList in group_to_docList.items():
            print_log("group key : ", key , "set value : ", group_value[key], "\n----for docList : ", docList)

            # 确保 doc_id 列存在，并且 columns[0] 也有效
            if 'doc_id' not in now_table.columns:
                raise ValueError("Column 'doc_id' not found in the DataFrame.")

            if columns[0] not in now_table.columns:
                raise ValueError(f"Column '{columns[0]}' not found in the DataFrame.")
            
            now_table.loc[now_table['doc_id'].isin(docList), columns[0]] = group_value[key]
        #print_log("group most value df : \n", df)

        #now_table = table_util.merge_table(now_table, df, key='doc_id')

        # step2-2 : query input, build the input and the query in LLM

        print_log("group final df : \n", now_table)


        self.output.append(TablePack(self.table, now_table))
        
        return None
        