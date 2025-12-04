from .pack import Pack

class TablePack(Pack):
    """
    tablepack includes table(df), tablename
    data is a dataframe
    """
    def __init__(self, tablename, table):
        super().__init__('Table')
        self.tablename = tablename
        self.table = table
          
    def parse(self):
        """
        Get main data
        """
        return self.table


