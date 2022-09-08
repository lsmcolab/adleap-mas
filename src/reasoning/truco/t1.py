#####
# TRUCO 1 ALGORITHM
#####
# returns a random action
def t1_planning(state, agent):
    all_none = True
    for card in agent.hand:
        if card is not None:
            all_none = False
    if all_none:
        return None,None

    action = state.action_space.sample()
    if len(agent.hand) != 0:
        while agent.hand[action] is None:
            action = state.action_space.sample()
    return action, None