#####
# TRUCO 2 ALGORITHM
#####
# returns the action with highest value (card value)
def t2_planning(state, agent):

    highest_card_index = None
    for i in range(len(agent.hand)):
        if agent.hand[i] is not None:
            if highest_card_index is None:
                highest_card_index = i
            elif agent.hand[highest_card_index][0] < agent.hand[i][0]:
                highest_card_index = i

    return highest_card_index, None