from src.reasoning.node import ANode, ONode
import random
import time
from src.reasoning.estimation import parameter_estimation

class POMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.max_depth = max_depth
        self.max_it = max_it
        self.c = 0.5
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        ###
        # POMCP enhancements
        ###
        # particle Revigoration (silver2010pomcp)
        particle_revigoration = kwargs.get('particle_revigoration')
        if particle_revigoration is not None:
            self.pr = particle_revigoration
        else: #default
            self.pr = True

        k = kwargs.get('k') # particle filter size
        self.k = k if k is not None else 100

        ###
        # Further settings
        ###
        stack_size = kwargs.get('state_stack_size')
        if stack_size is not None:
            self.state_stack_size = stack_size
        else: #default
            self.state_stack_size = 1

        ###
        # Evaluation
        ###
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        next_node = ANode(action,next_state,node.depth+1,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        return random.choice(state.get_actions_list())

    def rollout(self,node):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        node.depth += 2

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return reward +\
            self.discount_factor*self.rollout(node)

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return ONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth + 1:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node):
        # 1. Checking the stop condition
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node)

        self.simulation_count += 1
        start_t = time.time()
        
        # 3. Selecting the best action
        action = node.select_action(coef=self.c)

        # 4. Simulating the action
        (action_node, reward) = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 6. Getting the observation and adding the observation child on the tree
        observation_node = None
        observation = action_node.state.get_observation()
        
        for child in action_node.children:
            if child.observation.observation_is_equal(observation):
                observation_node = child
                observation_node.particle_filter.append(action_node.state)
                break
        
        if observation_node is None:
            observation_node = action_node.add_child(action_node.state,observation)
            observation_node.particle_filter.append(observation_node.state)

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 7. Calculating the reward, quality and updating the node
        R = reward + float(self.discount_factor * self.simulate(observation_node))
        node.particle_filter.append(node.state)
        node.update(action, R)
        observation_node.visits += 1
        return R

    def search(self, node, agent):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        while it < self.max_it:
            # a. Sampling the belief state for simulation
            if len(node.particle_filter) == 0:
                beliefState = node.state.sample_state(agent)
            else:
                beliefState = random.sample(node.particle_filter,1)[0]
            node.state = beliefState

            # b. simulating
            self.simulate(node)

            it += 1

        return node.get_best_action()

    def particle_revigoration(self,env,agent,root):
        # 1. Copying the current root particle filter
        current_particle_filter = []
        for particle in root.particle_filter:
            current_particle_filter.append(particle)
        
        # 2. Reinvigorating particles for the new particle filter or
        # picking particles from the uniform distribution
        root.particle_filter = []
        if len(current_particle_filter) > 0: # particle ~ F_r
            while(len(root.particle_filter) < self.k):
                particle = random.sample(current_particle_filter,1)[0]
                root.particle_filter.append(particle)
        else: # particle ~ U
            while(len(root.particle_filter) < self.k):
                particle = env.sample_state(agent)
                root.particle_filter.append(particle)

    def find_new_root(self,current_state,previous_action,current_observation,previous_root):
        # 1. If the root doesn't exist yet, create it
        # - NOTE: The root is always represented as an "observation node" since the next node
        # must be an action node.
        if previous_root is None:
            new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
            return new_root

        # 2. Else, walk on the tree to find the new one (giving the previous information)
        action_node, observation_node, new_root = None, None, None

        # a. walking over action nodes
        for child in previous_root.children:
            if child.action == previous_action:
                action_node = child
                break

        # - if we didn't find the action node, create a new root
        if action_node is None:
            new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
            return new_root

        # b. walking over observation nodes
        for child in action_node.children:
            if child.observation.observation_is_equal(current_observation):
                observation_node = child
                break

        # - if we didn't find the action node, create a new root
        if observation_node is None:
            new_root = ONode(observation=None,state=current_state,depth=0,parent=None)
            return new_root

        # 3. Definig the new root and updating the depth
        new_root = observation_node
        new_root.parent = None
        new_root.update_depth(0)
        return new_root

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if 'search_tree' not in agent.smart_parameters:
            root_node = ONode(observation=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            root_node = self.find_new_root(state, previous_action, current_observation, agent.smart_parameters['search_tree'])
            # if no valid node was found, reset the tree
            if root_node is None:
                root_node = ONode(observation=None,state=state,depth=0,parent=None)
        
        # 3. Estimating the parameters 
        if 'estimation_method' in agent.smart_parameters:
            root_node.state, agent.smart_parameters['estimation'] = parameter_estimation(root_node.state,agent,\
                agent.smart_parameters['estimation_method'], *agent.smart_parameters['estimation_args'])

        # 4. Performing particle revigoration
        if self.pr:
            self.particle_revigoration(state,agent,root_node)

        # 5. Searching for the best action within the tree
        best_action = self.search(root_node, agent)

        # 6. Returning the best action
        #root_node.show_qtable()
        
        return best_action, root_node, {'nrollouts': self.rollout_count,'nsimulations':self.simulation_count}

def write_stat(method):
    with open('results/rolloutxsimulation_pomcp.csv','a') as file:
        method.rollout_count = method.rollout_count if method.rollout_count != 0 else 1
        method.simulation_count = method.simulation_count if method.simulation_count != 0 else 1
        file.write(str(method.rollout_count)+';'+str(method.rollout_total_time)+';'+str(method.rollout_total_time/method.rollout_count)\
            +';'+str(method.simulation_count)+';'+str(method.simulation_total_time)+';'+str(method.simulation_total_time/method.simulation_count)+'\n')

def pomcp_planning(env, agent, max_depth=20, max_it=200, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    pomcp = POMCP(max_depth, max_it, kwargs)
    next_action, search_tree, info = pomcp.planning(copy_env,agent)

    # 3. Updating the search tree
    #write_stat(pomcp)
    agent.smart_parameters['search_tree'] = search_tree
    agent.smart_parameters['count'] = info
    return next_action,None