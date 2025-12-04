from abc import ABCMeta,abstractmethod

class Base(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        self.name = 'Base'
        self.input = []
        self.output = []
        self.additionDict = {} # addition data
        self.visited = 0 # 0 = unvisited, 1= visiting, 2=visited
        self.indexer = None

    def set_indexer(self, x):
        self.indexer = x

    def set_unvisited(self):
        self.visited = 0

    def set_input(self, x):
        self.input = x

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output
    
    def get_addtion(self):
        return self.additionDict

    def append_input(self, x):
        self.input.append(x)

    def extend_input(self, x):
        self.input.extend(x)

    def append_output(self, x):
        self.output.append(x)
    
    def extend_output(self, x):
        self.output.extend(x)

    def add_info(self, key, val):
        self.additionDict.setdefault(key, val)

class Logical(Base):
    def __init__(self):
        super().__init__()


class Physical(Base):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def process(self):
        pass