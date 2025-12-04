

class Processer(object):
    """
    Processer is the base class for all processers.
    It provides a common interface for processing SQL queries.
    """

    def __init__(self):
        pass

    def dfs(self, node):
        """
        Perform a depth-first search on the node.
        """
        node.visited = True
        for v in node.input:
            if not v.visited:
                self.dfs(v)
        
        # Process the node
        return node.process()

    def process(self, root):
        """
        Process the SQL query represented by the root node.
        """
        return self.dfs(root)