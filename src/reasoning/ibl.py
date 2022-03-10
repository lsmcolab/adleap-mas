from src.reasoning.node import IANode, IONode
from src.reasoning.qlearn import entropy
import random

def simulate_action(node, action, state_batch):
    # 1. Copying the current state for simulation
    tmp_state = node.state.copy()

    # 2. Acting
    reward = 0
    for i in range(state_batch):
        next_state,par_reward, _, _ = tmp_state.step(action)
        reward += par_reward
    next_state = tmp_state
    next_node = IANode(action,next_state,node.depth+1,node)

    # 3. Returning the next node and the reward
    return next_node, reward

def rollout_policy(state):
    return state.action_space.sample()

def rollout(node,max_depth,discount_factor,state_batch):
    # 1. Checking if it is an end state or leaf node
    if is_terminal(node) or is_leaf(node,max_depth):
        return 0

    # 2. Choosing an action
    tmp_state = node.state.copy()
    action = rollout_policy(tmp_state)

    # 3. Simulating the action
    reward = 0
    for i in range(state_batch):
        next_state,par_reward, _, _ = tmp_state.step(action)
        reward += par_reward
    next_state = tmp_state

    action_node = IANode(action,next_state,node.depth+1,node)
    observation = action_node.state.get_observation()
    observation_node = action_node.add_child(action_node.state,observation)

    # 4. Rolling out
    return reward +\
     discount_factor*rollout(observation_node,max_depth,discount_factor,state_batch)

def is_leaf(node, max_depth):
    if node.depth >= max_depth + 1:
        return True
    return False

def is_terminal(node):
    return node.state.state_set.is_final_state(node.state.state)

def simulate(node, max_depth, state_batch,discount_factor=0.95):
    # 1. Checking the stop condition
    node.visits += 1
    if is_leaf(node,max_depth):
        return 0

    # 2. Checking child nodes
    if node.children == []:
        # a. adding the children
        for action in node.actions:
            (next_node, reward) = simulate_action(node, action, state_batch)
            node.add_child(next_node.state,next_node.action)
        return rollout(node,max_depth,discount_factor, state_batch)

    # 3. Selecting the best action
    father = node.parent
    Px = 1 if father is None else (node.visits)/(father.visits+1)
    action = node.select_action(gamma=Px,mode='ibl')

    # 4. Simulating the action
    (action_node, reward) = simulate_action(node, action, state_batch)

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

    # Updating the particle filter set and entropy
    observation_node.update_filterset(action_state)
    observation_node.update_entropy()

    # 7. Calculating the reward, quality and updating the node
    R = reward + float(discount_factor * simulate(observation_node,max_depth,state_batch,discount_factor))
    node.update(action, R)
    return R

def search(node, agent, max_it, max_depth, state_batch):
    # 1. Performing the Monte-Carlo Tree Search
    it = 0
    while it < max_it:
        # a. Sampling the belief state for simulation
        if len(node.particle_filter) == 0:
            global SAMPLE_A, SAMPLE_T, N_SAMPLE
            beliefState = node.state.sample_state(agent)
        else:
            beliefState = random.sample(node.particle_filter,1)[0]

        # beliefState.simulator.draw_map()
        node.state = beliefState

        # b. simulating
        simulate(node, max_depth, state_batch)

        it += 1
    return node.get_best_action()

def black_box_update(env,agent,root,Px,k=100):
    # 1. Getting real-world current observation
    real_obs = env.get_observation()

    # 2. Updating the root particle filter
    new_particle_filter = []
    for particle in root.particle_filter:
        if particle.observation_is_equal(real_obs):
            new_particle_filter.append(particle)

    # 3. Reseting the particle filter
    root.update_entropy()
    root.particle_filter = [] 
    root.particles_set = {}
    
    if len(new_particle_filter) > 0:
        while len(root.particle_filter) < int(Px * k)+1:
            particle = random.sample(new_particle_filter,1)[0]
            root.update_filterset(particle)

    # 4. Sampling new particles while don't get k particles into the filter
    while(len(root.particle_filter) < (int(1 - root.entropy)*k)) and\
     len(root.particle_filter) < k:
        particle = env.sample_state(agent)
        root.update_filterset(particle)
    root.update_entropy()

def find_new_root(current_state,previous_action,current_observation,previous_root):
    # 1. If the root doesn't exist yet, create it
    # - NOTE: The root is always represented as an "observation node" since the next node
    # must be an action node.
    if previous_root is None:
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root, 0

    # 2. Else, walk on the tree to find the new one (giving the previous information)
    action_node, observation_node, new_root = None, None, None

    # a. walking over action nodes
    for child in previous_root.children:
        if child.action == previous_action:
            action_node = child
            break

    # - if we didn't find the action node, create a new root
    if action_node is None:
        #print('Action Node not Found')
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root, 0

    # b. walking over observation nodes
    Px = 0
    for child in action_node.children:
        for particle in child.particle_filter:
            if particle.observation_is_equal(current_observation):
                observation_node = child
                Px = (observation_node.visits)/(action_node.visits+1)
                break

    # - if we didn't find the action node, create a new root
    if observation_node is None:
        #print('Obs Node not Found')
        new_root = IONode(observation=None,state=current_state,depth=0,parent=None)
        return new_root, 0

    # 3. Definig the new root and updating the depth
    #print('New Root Found')
    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root, Px

def monte_carlo_planning(state, agent, max_it, max_depth,state_batch):
    # 1. Getting the current state and previous action-observation pair
    previous_action = agent.next_action
    current_observation = state.get_observation()

    # 2. Defining the root of our search tree
    # via initialising the tree
    if 'search_tree' not in agent.smart_parameters:
        Px = 0
        root_node = IONode(observation=None,state=state,depth=0,parent=None)
    # or advancing within the existent tree
    else:
        root_node, Px = find_new_root(state, previous_action, current_observation, agent.smart_parameters['search_tree'])
        # if no valid node was found, reset the tree
        if root_node is None:
            root_node = IONode(observation=None,state=state,depth=0,parent=None)

    # 3. Black-box updating
    black_box_update(state,agent,root_node, Px)

    # 4. Searching for the best action within the tree
    best_action = search(root_node, agent, max_it, max_depth,state_batch)

    # 5. Returning the best action
    #root_node.show_qtable()
    return best_action, root_node


def ibl_planning(env, agent, max_depth=20, max_it=100, state_batch=1):
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    next_action, search_tree =\
     monte_carlo_planning(copy_env,agent,max_it,max_depth,state_batch)

    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    return next_action, None