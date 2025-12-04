from abc import ABCMeta,abstractmethod

class Pack(object):
    __metaclass__ = ABCMeta
    def __init__(self, type):
        """
        type : Table, Text, Photo
        """
        self.type = type
        
    @abstractmethod
    def parse(self):
        """
        Get main data
        """
        pass
