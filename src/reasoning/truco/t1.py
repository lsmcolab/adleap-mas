#####
# TRUCO 1 ALGORITHM
#####
# returns a random action
def t1_planning(state, agent):
    action = state.action_space.sample()
    while None in state.components['player'][state.current_player].hand[action]:
        action = state.action_space.sample()
    return action, None