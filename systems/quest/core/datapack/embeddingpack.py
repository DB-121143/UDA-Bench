from .pack import Pack

class EmbeddingPack(Pack):
    """
    EmbeddingPack includes: doc_id, embedding
    """
    def __init__(self, doc_id, embedding, column):
        super().__init__('Embedding')
        self.doc_id = doc_id
        self.embedding = embedding
        self.column = column
        
    def parse(self):
        """
        Get main data
        """
        return self.embedding

class EmbeddingListPack(Pack):
    """
    textpack includes: doc_id, embeddingList, column
    """
    def __init__(self, doc_id, embeddingList, column):
        super().__init__('EmbeddingList')
        self.doc_id = doc_id
        self.embeddingList = embeddingList
        self.column = column
        
    def parse(self):
        """
        Get main data
        """
        return self.embeddingList
    
class EmbeddingDictPack(Pack):
    """
    textpack includes: doc_id, embeddingDict: {column1 : [text1, text2, ..]}
    """
    def __init__(self, doc_id, embeddingDict):
        super().__init__('EmbeddingDict')
        self.doc_id = doc_id
        self.embeddingDict = embeddingDict
        
    def parse(self):
        """
        Get main data
        """
        return self.embeddingDict
