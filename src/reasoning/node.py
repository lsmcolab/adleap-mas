from qlearn import create_qtable, uct_select_action
import random

class Node(object):

    def __init__(self, state, depth, parent=None):
        self.state = state
        self.depth = depth
        self.parent = parent
        self.children = []
        self.visits = 0 

    def add_child(self,state):
        child = Node(state,self.depth+1,self)
        self.children.append(child)
        return child

    def remove_child(self,child):
        for c in self.children:
            if c == child:
                self.children.remove(child)
                break

    def update_depth(self,new_depth):
        self.depth = new_depth
        for c in self.children:
            c.update_depth(new_depth+1)

class QNode(Node):

    def __init__(self,action, state, depth, parent=None):
        super(QNode,self).__init__(state,depth,parent)
        self.value = 0
        self.action = action
        self.actions = [i for i in range(state.action_space.n)]
        self.qtable = create_qtable(self.actions)

    def update(self, action, result):
        self.qtable[action]['trials'] += 1
        self.qtable[action]['sumvalue'] += result
        self.qtable[action]['qvalue'] += \
            (float(result) - self.qtable[action]['qvalue']) / float(self.qtable[action]['trials'])

    def select_action(self):
        return uct_select_action(self)

    def get_best_action(self):
        # 1. Intialising the support variables
        tieCases = []
        best_action, maxQ = None, -100000000000

        # 2. Looking for the best action (max qvalue action)
        for a in self.actions:
            if self.qtable[a]['qvalue'] > maxQ\
             and self.qtable[a]['trials'] > 0:
                maxQ = self.qtable[a]['qvalue']
                best_action = a

        # 3. Checking if a tie case exists
        for a in self.actions:
            if self.qtable[a]['qvalue'] == maxQ:
                tieCases.append(a)

        if len(tieCases) > 0:
            best_action = random.sample(tieCases,1)[0]
        # 4. Returning the best action
        if(best_action==None):
            print("SAD")
            best_action = random.sample(self.actions,1)[0]
        return best_action

    def show_qtable(self):
        print('%8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials'))
        for a in self.actions:
            print('%8s %4.4f %4.4f %8f' % (self.state.action_dict[a],self.qtable[a]['qvalue'],self.qtable[a]['sumvalue'],self.qtable[a]['trials']))
        print('-----------------')

class ANode(QNode):

    def __init__(self,action, state, depth, parent=None):
        super(ANode,self).__init__(action,state,depth,parent)
        self.particle_filter = []
        self.action = action
        self.observation = None

    def add_child(self,state,observation):
        child = ONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class ONode(QNode):

    def __init__(self,observation, state, depth, parent=None):
        super(ONode,self).__init__(None,state,depth,parent)
        self.particle_filter = []
        self.action = None
        self.observation = observation

    def add_child(self,state,action):
        child = ANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child