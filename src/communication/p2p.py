def get_broadcast():
    return P2P()

class P2P(object):

    target_options = ['closest','furthest']

    def __init__(self,target='closest'):
        self.protocol = 'p2p'
        self.max_distance = 20
        self.visibility = 'private'
        self.target = 'closest'
        self.cost = 0.02
