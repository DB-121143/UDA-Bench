from .retrieve import Retrieve
from quest.utils import *
from quest.core.datapack import *
import quest.conf.settings as settings


class RetrieveText(Retrieve):
    """
    Info : columns - a list of ColumnExpr
            table - the retrieve tableName
            type - the retrieve type ('Photo'/'Text'/..)

            self.retreieveList
            self.indexer
            self.sampler
    Input : None
    """
    def __init__(self, columns, table, type):
        super().__init__(columns, table, type)
        self.name = 'RetrieveText'

    def process(self):
        
        # {doc_id1 : { column1 :[text1, text2, ...], }

        columns = column_util.parse_full(self.columns) # [U.a1, U.a2, V.a1]

        # Step1 : Get evidence

        query_text = self.sampler.get_evidence()
     
        # Step 2 : Get text List and Pack

        for id in self.retrieveList:
            nowDict = {}
            for column in columns:
                #nowDict.setdefault(column, self.indexer.get_relative_chunks_text(id, query_text[column], topk = settings.TOPK))
                nowDict.setdefault(column, self.indexer.get_relative_chunks_text_with_id(id, query_text[column], topk = settings.TOPK))
            self.output.append(TextDictPack(id, nowDict))



        

