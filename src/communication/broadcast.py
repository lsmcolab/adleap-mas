def get_broadcast():
    return Broadcast()

class Broadcast(object):

    def __init__(self):
        self.protocol = 'broadcast'
        self.max_distance = 50
        self.visibility = 'public'
        self.cost = 0.02
