from gym import spaces
import numpy as np

import sys
sys.path.append('src/reasoning')

from src.reasoning.estimation import truco_uniform_estimation
from src.envs.TrucoEnv import TrucoEnv
import time

###
# Main
###
env = TrucoEnv(players   = ['A','B','C','D'],
                reasoning = ['pomcp','pomcp','t2','t1'],visibility='partial')
state = env.reset()

while env.state['points'][0] < 12 and env.state['points'][1] < 12:

    info, done = None, False
    while not done:
        # Rendering the environment
        env.render(env,info=info)

        # Agent taking an action
        current_player = env.get_adhoc_agent()
        module = __import__(current_player.type)
        method = getattr(module, current_player.type+'_planning')

        if current_player.type == 'mcts' or current_player.type == 'pomcp':
            current_player.next_action, _ = method(state,current_player,estimation_algorithm=truco_uniform_estimation)
        else:
            current_player.next_action, _ = method(state,current_player)

        # Step on environment
        state, reward, done, info = env.step(current_player.next_action)
        
        # Verifying the end condition
        if done:
            break
        
    env.render(env,info=info)
    env.deal()
    state = env.get_observation()

env.close()
