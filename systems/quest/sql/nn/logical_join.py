from .base import Logical

class LogicalJoin(Logical):
    """
    Info :  join_type - a list of join type ['INNER', ..]
            join_oreder - a list of BinaryNode U.a1 == V.a2 by tuple, [U.a1 == V.a2 , Team.team == Player.teamname]
            type - the extract type ('Photo'/'Text'/..)
    Input : LogicalFilter / LogicalJoin / LogicalExtract ..
    """
    def __init__(self, join_type, join_order, type):
        super().__init__()
        self.join_type = join_type
        self.join_order = join_order
        self.type = type
        self.name = 'LogicalJoin'