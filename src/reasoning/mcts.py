from node import QNode

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
    next_node = QNode(action,next_state,node.depth+1,node)

    # 4. Rolling out
    sim_agent = next_node.state.get_adhoc_agent()
    return reward +\
     discount_factor*rollout(next_node,sim_agent,max_depth,discount_factor)

def get_state_with_estimated_values(state):
    adhoc_agent = state.get_adhoc_agent()
    for agent in state.components['agents']:
        if agent.index != adhoc_agent.index:
            # print (agent.smart_parameters['estimations'].estimation_histories)
            # print (agent.index)

            selected_type = agent.smart_parameters['estimations'].get_highest_type_probability()
            selected_parameter = agent.smart_parameters['estimations'].get_parameters_for_selected_type(selected_type)
            # print (selected_type)
            # print (selected_parameter)
            agent.type= selected_type
            agent.angle= selected_parameter.angle
            agent.radius = selected_parameter.radius
            agent.level= selected_parameter.level

def simulate_action(node, agent, action):
    # 1. Copying the current state for simulation
    tmp_state = node.state.copy()

    # 2. Acting
    get_state_with_estimated_values(tmp_state)
    next_state,reward, _, _ = tmp_state.step(action)
    next_node = QNode(action,next_state,node.depth+1,node)

    # 3. Returning the next node and the reward
    return next_node, reward

def is_leaf(node, max_depth):
    if node.depth >= max_depth + 1:
        return True
    return False

def is_terminal(node):
    return node.state.state_set.is_final_state(node.state.state)

def simulate(node, agent, max_depth,discount_factor=0.9):
    # 1. Checking the stop condition
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
    (next_node, reward) = simulate_action(node, agent, action)

    # 5. Adding the child on the tree
    if next_node.action in [c.action for c in node.children]:
        for child in node.children:
            if next_node.action == child.action:
                next_node = child
                break
    else:
        node.children.append(next_node)

    # 6. Calculating the reward, quality and updating the node
    sim_agent = next_node.state.get_adhoc_agent()
    R = reward + float(discount_factor * simulate(next_node,sim_agent,max_depth,discount_factor))
    node.visits += 1
    node.update(action, R)
    return R

def monte_carlo_tree_search(state, action_space, agent, max_it, max_depth,estimation_algorithm):
    # 1. Defining the root node
    root_node = None
    if 'search_tree' not in agent.smart_parameters:
        root_node = QNode(action=None,
                        state=state,depth=0,parent=None)
    else:
        for c in agent.smart_parameters['search_tree'].children:
            if state.state_is_equal(c.state.state):
                root_node = c
                root_node.parent = None
                root_node.update_depth(0)
                break

    # 2. Checking if the root_node was defined
    if root_node is None:
        root_node = QNode(action=None,
                            state=state,depth=0,parent=None)

    # - estimating enviroment parameters
    # if estimation_algorithm is not None:
    #     root_node.state = estimation_algorithm(root_node.state)

    # - cleaning the memory cache
    import gc
    gc.collect()

    # 3. Performing the Monte-Carlo Tree Search
    it = 0
    while it < max_it:
        simulate(root_node, agent, max_depth)
        it += 1

    # 4. Retuning the best action and the search tree root node
    return root_node.get_best_action(),root_node

def mcts_planning(env,agent,max_depth=10, max_it=100,estimation_algorithm=None):
    # 1. Setting the environment for simulation

    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True
    # 2. Planning
    next_action, search_tree =\
     monte_carlo_tree_search(copy_env,copy_env.action_space,agent,max_it,max_depth,estimation_algorithm)

    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree

    return next_action, None