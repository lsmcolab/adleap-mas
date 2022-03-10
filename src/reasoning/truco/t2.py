#####
# TRUCO 2 ALGORITHM
#####
# returns the action with highest value (card value)
def t2_planning(state, agent):

    highest_card_index = None
    for i in range(len(state.components['player'][state.current_player].hand)):
        if highest_card_index is not None and None not in state.components['player'][state.current_player].hand[i]:
            if state.components['player'][state.current_player].hand[highest_card_index][0] < \
             state.components['player'][state.current_player].hand[i][0]:
                highest_card_index = i
        elif None not in state.components['player'][state.current_player].hand[i]:
            highest_card_index = i
    return highest_card_index, None