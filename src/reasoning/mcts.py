from src.reasoning.node import QNode

def rollout_policy(state):
    return state.action_space.sample()

def rollout(node,max_depth,discount_factor):
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
    return reward +\
     discount_factor*rollout(next_node,max_depth,discount_factor)

def simulate_action(node, action):
    # 1. Copying the current state for simulation
    tmp_state = node.state.copy()

    # 2. Acting
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

def simulate(node, max_depth,discount_factor=0.9):
    # 1. Checking the stop condition
    if is_terminal(node) or is_leaf(node,max_depth):
        return 0

    # 2. Checking child nodes
    if node.children == []:
        # a. adding the children
        for action in node.actions:
            (next_node, reward) = simulate_action(node, action)
            node.children.append(next_node)
        return rollout(node,max_depth,discount_factor)

    # 3. Selecting the best action
    action = node.select_action()

    # 4. Simulating the action
    (next_node, reward) = simulate_action(node, action)

    # 5. Adding the child on the tree
    if next_node.action in [c.action for c in node.children]:
        for child in node.children:
            if next_node.action == child.action:
                next_node = child
                break
    else:
        node.children.append(next_node)

    # 6. Calculating the reward, quality and updating the node
    R = reward + float(discount_factor * simulate(next_node,max_depth,discount_factor))
    node.visits += 1
    node.update(action, R)
    return R

def monte_carlo_tree_search(state, agent, max_it, max_depth,estimation_algorithm):
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
        
    #####
    # ESTIMATION METHOD UPDATE: START
    #####
    # - performing estimation
    if estimation_algorithm is not None and 'estimation_args' in agent.smart_parameters:
        root_node.state, agent.smart_parameters['estimation'] = \
            estimation_algorithm(root_node.state,agent,*agent.smart_parameters['estimation_args'])
    elif estimation_algorithm is not None and 'estimation_args' not in agent.smart_parameters:
        root_node.state, agent.smart_parameters['estimation'] = estimation_algorithm(root_node.state,agent)
    else:
        from src.reasoning.estimation import uniform_estimation
        root_node.state = uniform_estimation(root_node.state)
    #####
    # ESTIMATION METHOD UPDATE: END
    #####

    # - cleaning the memory cache
    import gc
    gc.collect()

    # 3. Performing the Monte-Carlo Tree Search
    it = 0
    while it < max_it:
        # - estimating environment parameters
        if estimation_algorithm is not None:
            root_adhoc_agent = root_node.state.get_adhoc_agent()
            root_adhoc_agent.smart_parameters['estimation'] = agent.smart_parameters['estimation']
            root_node.state = root_adhoc_agent.smart_parameters['estimation'].sample_state(state)

        # - simulating
        simulate(root_node, max_depth)
        it += 1

    # 4. Retuning the best action and the search tree root node
    return root_node.get_best_action(), root_node

def mcts_planning(env,agent,max_depth=20, max_it=100,estimation_algorithm=None):
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.viewer = None
    copy_env.simulation = True

    # 2. Planning
    next_action, search_tree =\
     monte_carlo_tree_search(copy_env,agent,max_it,max_depth,estimation_algorithm)

    # 3. Updating the search tree
    agent.smart_parameters['search_tree'] = search_tree
    return next_action, None