from .pack import Pack

class DocListPack(Pack):
    """
    textpack includes:  text, column
    """
    def __init__(self, tablename, lst):
        super().__init__('Text')
        self.tablename = tablename
        self.docList = lst
        
    def parse(self):
        """
        Get main data
        """
        return self.docList