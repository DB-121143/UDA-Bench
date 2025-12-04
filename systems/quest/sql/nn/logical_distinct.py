from .base import Logical

class LogicalDistinct(Logical):
    """
    Info :  
    Input : L
    """
    def __init__(self):
        super().__init__()
        self.name = 'LogicalDistinct'