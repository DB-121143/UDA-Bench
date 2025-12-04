from .retrieve import Retrieve
from quest.utils import *
from quest.core.datapack import *
import quest.conf.settings as settings
from quest.utils.log import print_log

class RetrieveEmbedding(Retrieve):
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
        self.name = 'RetrieveEmbedding'

    def set_querier(self, x):
        self.querier = x

    def process(self):
        
        # {doc_id1 : { column1 :[text1, text2, ...], }

        columns = column_util.parse_full(self.columns) # [U.a1, U.a2, V.a1] group by columns

        # Step1 : Get evidence

        query_text = self.sampler.get_evidence()
     
        # Step 2 : Get text List and Pack

        if self.type == 'Chunk':
            # chunk or sentences
            for id in self.retrieveList:
                for column in columns:
                    results = self.indexer.get_relative_chunks_text_with_id_and_embedding(id, query_text[column], topk = 2)
                    now_embedding = [result[2] for result in results] 
                    self.output.append(EmbeddingListPack(id, now_embedding, column))
        else:
            # doc 
            print_log("get Doc Embedding !! -- ", self.retrieveList)
            for id in self.retrieveList:
                column = self.columns[0]
                now_embedding = self.indexer.get_doc_embedding(id)
                self.output.append(EmbeddingPack(id, now_embedding, column))

                



        

