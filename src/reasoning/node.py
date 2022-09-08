from src.reasoning.qlearn import create_qtable, uct_select_action, ibl_select_action, entropy
import math
import random

"""
    Traditional tree search nodes
"""
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

"""
    Quality search nodes
"""
class QNode(Node):

    def __init__(self,action, state, depth, parent=None):
        super(QNode,self).__init__(state,depth,parent)
        self.value = 0
        self.action = action
        self.actions = state.get_actions_list()
        self.qtable = create_qtable(self.actions)

    def update(self, action, result):
        self.visits += 1
        self.qtable[str(action)]['trials'] += 1
        self.qtable[str(action)]['sumvalue'] += result
        self.qtable[str(action)]['qvalue'] += \
            (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])

        self.value += (result-self.value)/self.visits

    def select_action(self,coef=0.5,mode='uct'):
        if mode == 'ibl':
            return ibl_select_action(self,alpha=coef)
        else:
            return uct_select_action(self,c=coef)

    def get_best_action(self):
        # 1. Intialising the support variables
        tieCases = []
        best_action, maxQ = None, -100000000000

        # 2. Looking for the best action (max qvalue action)
        for a in self.actions:
            if self.qtable[str(a)]['qvalue'] > maxQ\
             and self.qtable[str(a)]['trials'] > 0:
                maxQ = self.qtable[str(a)]['qvalue']
                best_action = a

        # 3. Checking if a tie case exists
        for a in self.actions:
            if self.qtable[str(a)]['qvalue'] == maxQ:
                tieCases.append(a)

        if len(tieCases) > 0:
            best_action = random.sample(tieCases,1)[0]

        # 4. Returning the best action
        if(best_action==None):
            best_action = random.sample(self.actions,1)[0]
            
        return best_action

    def show_qtable(self):
        print('%8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1]), reverse=True)
        
        for a in action_dict:
            print('%8s %4.4f %4.4f %8f' % (self.state.action_dict[a],self.qtable[str(a)]['qvalue'],\
                                        self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')

class ANode(QNode):

    def __init__(self,action, state, depth, parent=None):
        super(ANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,state,observation):
        child = ONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class ONode(QNode):

    def __init__(self,observation, state, depth, parent=None):
        super(ONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.particles_set = {}
        
    def add_child(self,state,action):
        child = ANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

"""
    IB-POMCP nodes
"""
class IANode(ANode):

    def __init__(self,action, state, depth, parent=None):
        super(IANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,state,observation):
        child = IONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class IONode(ONode):

    def __init__(self,observation, state, depth, parent=None):
        super(IONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.particles_set = {}
        self.cum_entropy = 0
        self.max_entropy = 1
        
    def add_child(self,state,action):
        child = IANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def get_alpha(self):
        if self.visits == 0:
            return 1
        else:
            cum_max_entropy = self.visits*self.max_entropy
            entropy_rate_function = (self.cum_entropy/cum_max_entropy)
            visit_decay_function = math.e*math.log(self.visits)/self.visits
            return visit_decay_function*entropy_rate_function

    def update(self, action, result):
        self.visits += 1
        self.qtable[str(action)]['trials'] += 1

        self.qtable[str(action)]['sumvalue'] += result

        self.qtable[str(action)]['qvalue'] += \
            (float(result) - self.qtable[str(action)]['qvalue']) / float(self.qtable[str(action)]['trials'])
        self.value += (result-self.value)/self.visits


    def update_entropy(self,obs_state):
        # adding the state to the particle set
        hash_key = hash(str(obs_state.state))
        if hash_key in self.particles_set:
            self.particles_set[hash_key] += 1
        else:
            self.particles_set[hash_key] = 1
            
        # calculating the new entropy and updating the
        # maximum entropy
        new_entropy = entropy(self.particles_set)
        if new_entropy > self.max_entropy:
            self.max_entropy = new_entropy

        # normalising the new entropy
        self.cum_entropy += abs(new_entropy)

"""
    rho-POMCP nodes
"""
class RhoANode(ANode):

    def __init__(self,action, state, depth, parent=None):
        super(RhoANode,self).__init__(action,state,depth,parent)
        self.action = action
        self.observation = None

    def add_child(self,state,observation):
        child = RhoONode(observation,state,self.depth+1,self)
        self.children.append(child)
        return child

class RhoONode(ONode):

    def __init__(self,observation, state, depth, parent=None):
        super(RhoONode,self).__init__(None,state,depth,parent)
        self.action = None
        self.observation = observation
        
        self.particle_filter = []
        self.cummulative_bag = {}
    
    def add_child(self,state,action):
        child = RhoANode(action,state,self.depth+1,self)
        self.children.append(child)
        return child

    def add_to_cummulative_bag(self,particle,action):
        obs_p = particle.get_obs_p(action)[1]
        hash_key = hash(str(particle.state))
        if hash_key not in self.cummulative_bag:
            self.cummulative_bag[hash_key] =  [particle,obs_p]
        else:
            self.cummulative_bag[hash_key] =  [particle,self.cummulative_bag[hash_key][1] + obs_p]