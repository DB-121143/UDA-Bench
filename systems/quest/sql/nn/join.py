from .base import Physical

class Join(Physical):
    """
    Info : join_type - a list of join type ['INNER', ..]
            join_oreder - a list of equal order U.a1 = V.a2 by tuple, [(U.a1, V.a2) , (Team.team = Player.teamname)]
            type - the extract type ('Photo'/'Text'/..)
    Input : None
    """
    def __init__(self, join_type, join_order, type):
        super().__init__()
        self.name = 'Join'
        self.join_type = join_type
        self.join_oreder = join_order
        self.type = type

    def process(self):
        pass