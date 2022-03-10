#####
# TRUCO 3 ALGORITHM
#####
# returns the action with lowest value (card value)
def t3_planning(state, agent):
    lowest_card_index = None
    for i in range(len(state.components['player'][state.current_player].hand)):
        if lowest_card_index is not None and  None not in state.components['player'][state.current_player].hand[i]:
            if state.components['player'][state.current_player].hand[lowest_card_index][0] > \
             state.components['player'][state.current_player].hand[i][0]:
                lowest_card_index = i
        elif None not in state.components['player'][state.current_player].hand[i]:
            lowest_card_index = i
    return lowest_card_index, None