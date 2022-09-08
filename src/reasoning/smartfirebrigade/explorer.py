import numpy as np
import random as rd

def explorer_planning(env,agent):
    if env.action_mode == 'policy':
        return 2, None

    next_action = None
    pos, dir = agent.position, agent.direction
    checkpoints = [
        (int(env.dim[0]/4),int(env.dim[1]/4)),
        (int(3*env.dim[0]/4),int(env.dim[1]/4)),
        (int(3*env.dim[0]/4),int(3*env.dim[1]/4)),
        (int(env.dim[0]/4),int(3*env.dim[1]/4)),
    ]
    if agent.target is None:
        sampled =  rd.sample(checkpoints,1)
        agent.target = sampled[0]
        
    if pos[0] > agent.target[0]:
        return env.get_action('WEST'), agent.target
    elif pos[0] < agent.target[0]:
        return env.get_action('EAST'), agent.target  
    elif pos[1] > agent.target[1]:
        return env.get_action('SOUTH'), agent.target
    elif pos[1] < agent.target[1]:
        return env.get_action('NORTH'), agent.target
    else:
        agent.target = None, None
    return 0, None