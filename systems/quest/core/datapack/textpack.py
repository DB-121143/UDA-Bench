from .pack import Pack

class TextPack(Pack):
    """
    textpack includes: doc_id, text, column
    """
    def __init__(self, doc_id, text, column):
        super().__init__('Text')
        self.doc_id = doc_id
        self.text = text
        self.column = column
        
    def parse(self):
        """
        Get main data
        """
        return self.text

class TextListPack(Pack):
    """
    textpack includes: doc_id, textList, column
    """
    def __init__(self, doc_id, textList, column):
        super().__init__('TextList')
        self.doc_id = doc_id
        self.textList = textList
        self.column = column
        
    def parse(self):
        """
        Get main data
        """
        return self.textList
    
class TextDictPack(Pack):
    """
    textpack includes: doc_id, textDict: {column1 : [text1, text2, ..]}
    """
    def __init__(self, doc_id, textDict):
        super().__init__('TextDict')
        self.doc_id = doc_id
        self.textDict = textDict
        
    def parse(self):
        """
        Get main data
        """
        return self.textDict
