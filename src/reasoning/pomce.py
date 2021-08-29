# Implementation of POMCP based estimation
from src.reasoning.estimation import pomcp_estimation
import numpy as np
from node import ANode,ONode
import warnings
import random as rd
from estimation import uniform_estimation
warnings.filterwarnings("ignore")

# POMCP helper functions
def simulate_action(node, agent, action):
    # 1. Copying the current state for simulation
    tmp_state = node.state.copy()

    # 2. Acting
    next_state,reward, _, _ = tmp_state.step(action)
    next_node = ANode(action,next_state,node.depth+1,node)

    # 3. Returning the next node and the reward
    return next_node, reward

def rollout_policy(state):
    return state.action_space.sample()

def rollout(node,agent,max_depth,discount_factor):
    # 1. Checking if it is an end state or leaf node
    if is_terminal(node) or is_leaf(node,max_depth):
        return 0

    # 2. Choosing an action
    tmp_state = node.state.copy()
    action = rollout_policy(tmp_state)

    # 3. Simulating the action
    next_state, reward, _, _ = tmp_state.step(action)
    action_node = ANode(action,next_state,node.depth+1,node)
    observation = action_node.state.get_observation()
    observation_node = action_node.add_child(action_node.state,observation)

    # 4. Rolling out
    sim_agent = observation_node.state.get_adhoc_agent()
    return reward +\
    discount_factor*rollout(observation_node,sim_agent,max_depth,discount_factor)

def is_leaf(node, max_depth):
    if node.depth >= max_depth + 1:
        return True
    return False

def is_terminal(node):
    return node.state.state_set.is_final_state(node.state.state)

def simulate(node, agent, max_depth,discount_factor=0.9):
    # 1. Checking the stop condition
    node.particle_filter.append(node.state)
    if is_terminal(node) or is_leaf(node,max_depth):
        return 0

    # 2. Checking child nodes
    if node.children == []:
        # a. adding the children
        for action in node.actions:
            (next_node, reward) = simulate_action(node, agent, action)
            node.children.append(next_node)
        return rollout(node,agent,max_depth,discount_factor)

    # 3. Selecting the best action
    action = node.select_action()

    # 4. Simulating the action
    (action_node, reward) = simulate_action(node, agent, action)

    # 5. Adding the action child on the tree
    if action_node.action in [c.action for c in node.children]:
        for child in node.children:
            if action_node.action == child.action:
                action_node = child
                action_node.particle_filter.append(action_node.state)
                break
    else:
        node.children.append(action_node)
        action_node.particle_filter.append(action_node.state)

    # 6. Getting the observation and adding the observation child on the tree
    observation = action_node.state.get_observation()
    for child in action_node.children:
        for particle in child.particle_filter:
            if particle.observation_is_equal(observation):
                observation_node = child
                observation_node.particle_filter.append(action_node.state)
                break
    else:
        observation_node = action_node.add_child(action_node.state,observation)
        observation_node.particle_filter.append(observation_node.state)

    # 7. Calculating the reward, quality and updating the node
    sim_agent = observation_node.state.get_adhoc_agent()
    R = reward + float(discount_factor * simulate(observation_node,sim_agent,max_depth,discount_factor))
    node.visits += 1
    node.update(action, R)
    return R

def search(node, template_types,agent, max_it, max_depth):
    # 1. Performing the Monte-Carlo Tree Search
    it = 0
    while it < max_it:
        # a. Sampling the belief state for simulation
        if len(node.particle_filter) == 0:
            beliefState = uniform_estimation(node.state.sample_state(agent),template_types)
        else:
            beliefState = rd.sample(node.particle_filter,1)[0]

        # beliefState.simulator.draw_map()
        node.state = beliefState

        # b. simulating
        simulate(node, agent, max_depth)

        it += 1
    return node.get_best_action()

def black_box_update(env,template_types,agent,root,k=100):
    # 1. Getting real-world current observation
    real_obs = env.get_observation()

    # 2. Updating the root particle filter
    new_particle_filter = []
    for particle in root.particle_filter:
        if particle.observation_is_equal(real_obs):
            new_particle_filter.append(particle)
    root.particle_filter = new_particle_filter
    
    # 3. Sampling new particles while don't get k particles into the filter
    while(len(root.particle_filter) < k):
        sampled_env = env.sample_state(agent)
        sampled_env = uniform_estimation(sampled_env,template_types)
        root.particle_filter.append(sampled_env)

def find_new_root(current_state,previous_action,current_observation,previous_root):
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
        for particle in child.particle_filter:
            if particle.observation_is_equal(current_observation):
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

def monte_carlo_planning(state, action_space, template_types,agent, max_it, max_depth,estimation_algorithm,particles=100):
    # 1. Getting the current state and previous action-observation pair
    previous_action = agent.next_action
    current_observation = state.get_observation()

    # 2. Defining the root of our search tree
    if 'est_search_tree' not in agent.smart_parameters:
        root_node = ONode(observation=None,state=state,depth=0,parent=None)
    else:
        root_node = find_new_root(state, previous_action, current_observation, agent.smart_parameters['est_search_tree'])

    # 2. Checking if the root_node was defined
    if root_node is None:
        root_node = ONode(observation=None,state=state,depth=0,parent=None)

    # - and estimating enviroment parameters

    from estimation import uniform_estimation
    if estimation_algorithm is None:
        root_node.state = uniform_estimation(root_node.state,template_types)
    else:
        root_node.state = estimation_algorithm(root_node.state)

    # 3. Black-box updating
    black_box_update(state,template_types,agent,root_node,particles)

    # 3. Searching for the best action within the tree
    best_action = search(root_node, template_types,agent, max_it, max_depth)
    #print("Particle : ", root_node.particle_filter)

    # 4. Returning the best action
    return best_action, root_node

def sample_estimate(env,agent=None):
    adhoc_agent = env.get_adhoc_agent()
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)
    
    return env

# Estimation Class 

class POMCE(object):
    def __init__(self,env,template_types,parameter_minmax,discount_factor=0.9,max_iter=100,max_depth=10,min_particles=100):
        # Initialising the POMCE parameters
        self.template_types = template_types
        self.nparameters = len(parameter_minmax)
        # TODO : Remove this if not useful
        self.parameters_minmax = parameter_minmax
        self.discount_factor = discount_factor
        self.max_iter = max_iter
        self.max_depth = max_depth
        # Minimum number of particles in black-box
        self.min_particles = min_particles
        self.teammate = {}
        self.root = None
        self.check_teammates_estimation_set(env)

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            tindex = teammate.index
            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}
                
                # for each type
                for type in self.template_types:
                    # create the estimation set and the bag of estimators
                    self.teammate[teammate.index][type] = {}
                    self.teammate[teammate.index][type]['probability_history'] = [1/len(self.template_types)]
                    self.teammate[teammate.index][type]['parameter_estimation_history'] = \
                        [np.array([rd.uniform(self.parameters_minmax[n][0],self.parameters_minmax[n][1]) for n in range(self.nparameters)])]

        return 

    # Question : Do we sample from our own estimation ? 
    
    def update(self,env):
        self.check_teammates_estimation_set(env)
        max_it =self.max_iter
        max_depth = self.max_depth
        env = env.copy()
        env.simulation = True
        env.viewer = None
        agent = env.get_adhoc_agent()

        next_action, root_node = monte_carlo_planning(env,env.action_space,self.template_types,agent,max_it,max_depth,sample_estimate,self.min_particles)

        # 3. Updating the search tree
        agent.smart_parameters['est_search_tree'] = root_node
        self.root_node = root_node
        
        for agent in env.components['agents']:
            if (agent.index == env.get_adhoc_agent().index):
                continue
            tindex = agent.index

            # To occount of PO : total occurence != len(root.particles)
            total_occurence = 0

            type_dict = {}
            for t in self.template_types:
                type_dict[t] = 0

            for sampled_env in root_node.particle_filter:
                for ag in sampled_env.components['agents']:
                    if ag.index == agent.index:
                        type_dict[ag.type]+=1
                        total_occurence += 1
                        break

            for t in self.template_types:
                if(type_dict[t]==0):
                    prev_est = self.teammate[agent.index][t]['parameter_estimation_history'][-1]
                    self.teammate[agent.index][t]['parameter_estimation_history'].append(prev_est)
                    self.teammate[agent.index][t]['probability_history'].append(0)
                    continue
                parameter = np.array([0 for i in range(0,self.nparameters)],dtype="float")
                
                for sampled_env in root_node.particle_filter:
                    for ag in sampled_env.components["agents"]:
                        if agent.index ==ag.index:
                            sampled_ag = ag
                            break
                    
                    if (sampled_ag.type == t):
                        parameter += sampled_ag.get_parameters()
                
                parameter = parameter/type_dict[t]
                
                
                self.teammate[agent.index][t]['parameter_estimation_history'].append(parameter)
                self.teammate[agent.index][t]["probability_history"].append(type_dict[t]/total_occurence)
        return self



    def get_estimation(self,env):
        type_probabilities, estimated_parameters = [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index not in self.teammate.keys() and teammate.index != adhoc_agent.index:
                type_prob = np.array([-1 for i in range(0,len(self.template_types))]) 
                parameter_est = []
                for type in self.template_types:
                    parameter_est.append(np.array([-1 for i in range(0,self.nparameters)]))
                type_probabilities.append(list(type_prob))
                estimated_parameters.append(parameter_est)
                continue

            if teammate.index != adhoc_agent.index:
                # type result
                type_prob = []
                for type in self.template_types:
                    type_prob.append(self.teammate[teammate.index][type]['probability_history'][-1])
                type_probabilities.append(list(type_prob))

                # parameter result
                parameter_est = []
                for type in self.template_types:
                    parameter_est.append(self.teammate[teammate.index][type]['parameter_estimation_history'][-1])
                estimated_parameters.append(list(parameter_est))

        return type_probabilities, estimated_parameters

    def sample_type_for_agent(self, teammate):
        type_prob = np.zeros(len(self.template_types))
        for i in range(len(self.template_types)):
            type = self.template_types[i]
            type_prob[i] = self.teammate[teammate.index][type]['probability_history'][-1]
        
        sampled_type = rd.choices(self.template_types,type_prob,k=1)
        return sampled_type[0]

    def get_parameter_for_selected_type(self, teammate, selected_type):
        parameter_est = self.teammate[teammate.index][selected_type]['parameter_estimation_history'][-1]
        return parameter_est            

    def sample_state(self, env):
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                selected_type = self.sample_type_for_agent(teammate)
                selected_parameter = self.get_parameter_for_selected_type(teammate,selected_type)

                teammate.type = selected_type
                teammate.set_parameters(selected_parameter)

        return env




