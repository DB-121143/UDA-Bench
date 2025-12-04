from .pack import Pack

class RetrievalQueryPack(Pack):
    """
    data is a chunk, can be parse by indexer
    """
    def __init__(self, data):
        super().__init__('Chunk', data)
        
    def parse(self):
        """
        Get data
        """
        return self.data
    
    def get_name(self):
        pass