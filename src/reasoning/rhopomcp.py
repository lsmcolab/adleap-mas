from cgitb import small
from src.reasoning.node import RhoANode, RhoONode
import random

def rhofunction(node, instant_reward):
    tmp_state = node.state.copy()
    for particle in node.particles_set:
        if tmp_state.state_is_equal(particle.state): 
            b_s = (node.particles_set[particle]/(node.visits+1)) 
            break
        else:
            b_s = 1/(node.visits+1)
    belief_reward = b_s*instant_reward
    return belief_reward

def simulate_action(node, action):
    # 1. Copying the current state for simulation
    tmp_state = node.state.copy()

    # 2. Acting
    next_state, reward, _, _ = tmp_state.step(action)
    next_node = RhoANode(action,next_state,node.depth+1,node)

    # 3. Returning the next node and the reward
    return next_node, rhofunction(node,reward)

def rollout_policy(state):
    return state.action_space.sample()

def rollout(node,smallbag,max_depth,discount_factor):
    # 1. Checking if it is an end state or leaf node
    if is_terminal(node) or is_leaf(node,max_depth):
        return 0

    # 2. Choosing an action
    tmp_state = node.state.copy()
    action = rollout_policy(tmp_state)

    # 3. Simulating the action
    next_state, reward, _, _ = tmp_state.step(action)
    action_node = RhoANode(action,next_state,node.depth+1,node)
    observation = action_node.state.get_observation()
    observation_node = action_node.add_child(action_node.state,observation)
    
    new_smallbag = smallbag_simulation(smallbag, action, observation)

    # 4. Rolling out
    return reward+\
     discount_factor*rollout(observation_node,new_smallbag,max_depth,discount_factor)

def is_leaf(node, max_depth):
    if node.depth >= max_depth + 1:
        return True
    return False

def is_terminal(node):
    return node.state.state_set.is_final_state(node.state.state) 

def smallbag_simulation(smallbag, action, observation):
    new_smallbag = []
    for i in range(len(smallbag)):
        sampled_state = random.sample(smallbag,1)[0]
        tmp_state = sampled_state.copy()

        # simulating the action
        next_state, r, _, _ = tmp_state.step(action)

        # checking the rejection condition
        if next_state.observation_is_equal(observation):
            new_smallbag.append(next_state)
    return new_smallbag

def simulate(node, max_depth, smallbag, discount_factor=0.95):
    # 1. Checking the stop condition
    node.particle_filter.append(node.state)
    for state in smallbag:
        node.update_filterset(state)

    if is_leaf(node,max_depth):
        return 0

    # 2. Checking child nodes
    if node.children == []:
        # a. adding the children
        for action in node.actions:
            (next_node, reward) = simulate_action(node, action)
            node.children.append(next_node)
        return rollout(node,smallbag,max_depth,discount_factor)

    # 3. Selecting the best action
    action = node.select_action()

    # 4. Simulating the action
    (action_node, reward) = simulate_action(node, action)

    # 5. Adding the action child on the tree
    action_state = action_node.state
    if action_node.action in [c.action for c in node.children]:
        for child in node.children:
            if action_node.action == child.action:
                action_node = child
                break
    else:
        action_node = node.add_child(action_state, action_node.action)
    action_node.visits += 1

    # 6. Getting the observation and adding the observation child on the tree
    observation = action_state.get_observation()
    for child in action_node.children:
        if child.observation.observation_is_equal(observation):
            observation_node = child
            break
    else:
        observation_node = action_node.add_child(action_state,observation)
    observation_node.observation = observation
    
    # Updating the particle filter set
    observation_node.update_filterset(action_state)

    # 7. Generating the new small bag
    new_smallbag = smallbag_simulation(smallbag, action, observation)

    # 8. Calculating the reward, quality and updating the node
    R = reward +\
         float(discount_factor * simulate(observation_node,max_depth,new_smallbag,discount_factor))
    node.visits += 1
    node.update(action, R)
    return R

def search(node, agent, max_it, max_depth, n):
    # 1. Performing the Monte-Carlo Tree Search
    it = 0
    while it < max_it:
        # a. Sampling the belief state for simulation
        if len(node.particle_filter) == 0:
            sampled_states = node.state.sample_nstate(agent, 1 + n)
            beliefState, smallbag = sampled_states[0], sampled_states[1:]
        else:
            sampled_states = random.sample(node.particle_filter, 1 + n)
            beliefState, smallbag = sampled_states[0], sampled_states[1:]

        # beliefState.simulator.draw_map()
        node.state = beliefState

        # b. simulating
        simulate(node, max_depth, smallbag)

        it += 1
    return node.get_best_action()

def black_box_update(env,agent,root,k=100):
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
        root.particle_filter.append(sampled_env)

def find_new_root(current_state,previous_action,current_observation,previous_root):
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the next node
    # must be an action node.
    if previous_root is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
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
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # b. walking over observation nodes
    for child in action_node.children:
        for particle in child.particle_filter:
            if particle.observation_is_equal(current_observation):
                observation_node = child
                break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        new_root = RhoONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root

    # 3. Definig the new root and updating the depth
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root

def monte_carlo_planning(state, agent, max_it, max_depth,smallbag_size):
    # 1. Getting the current state and previous action-observation pair
    previous_action = agent.next_action
    current_observation = state.get_observation()

    # 2. Defining the root of our search tree
    # via initialising the tree
    if 'search_tree' not in agent.smart_parameters:
        root_node = RhoONode(observation=None,state=state,depth=0,parent=None)
    # or advancing within the existent tree
    else:
        root_node = find_new_root(state, previous_action, current_observation, agent.smart_parameters['search_tree'])
        # if no valid node was found, reset the tree
        if root_node is None:
            root_node = RhoONode(observation=None,state=state,depth=0,parent=None)

    # 3. Black-box updating
    black_box_update(state,agent,root_node)

    # 4. Searching for the best action within the tree
    best_action = search(root_node, agent, max_it, max_depth, smallbag_size)

    # 5. Returning the best action
    #root_node.show_qtable()
    return best_action, root_node


def rhopomcp_planning(env, agent, max_depth=20, max_it=100, smallbag_size=10):
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    next_action, search_tree =\
     monte_carlo_planning(copy_env,agent,max_it,max_depth,smallbag_size)

    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    return next_action,None