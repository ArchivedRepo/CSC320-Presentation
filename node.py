class Node:
    """
    Node in graph search.
    """
    __slots__ = ['row', 'col', 'pred', 'cost']
    def __init__(self, x, y):
        self.row = x
        self.col = y
        self.pred = None
        self.cost = 0
    
    