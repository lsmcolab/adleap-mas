###
# IMPORTS
###
import os
import sys

sys.path.append(os.getcwd())

from src.reasoning.estimation import truco_uniform_estimation
from src.envs.TrucoEnv import TrucoEnv


###
# TRUCO CARD GAME ENVIRONMENT SETTINGS
###
env = TrucoEnv(players   = ['A','B','C','D'],
                reasoning = ['t1','t1','t2','t1'],visibility='partial',display=True)
state = env.reset()

###
# ADLEAP-MAS MAIN ROUTINE
###
while env.state['points'][0] < 12 and env.state['points'][1] < 12:

    info, done = None, False
    while not done:
        env.render(env,info=info)

        # 1. Importing agent method
        current_player = env.get_adhoc_agent()
        method = env.import_method(current_player.type)

        # 2. Reasoning about next action and target
        if current_player.type == 'mcts' or current_player.type == 'pomcp':
            current_player.next_action, _ = method(state,current_player,estimation_algorithm=truco_uniform_estimation)
        else:
            current_player.next_action, _ = method(state,current_player)

        # 3. Taking a step in the environment
        state, reward, done, info = env.step(current_player.next_action)
        
    env.render(env,info=info)
    env.deal()
    state = env.get_observation()

env.close()
###
# THE END - That's all folks :)
###