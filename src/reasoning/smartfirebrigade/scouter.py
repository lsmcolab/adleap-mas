import numpy as np
import random as rd

def scouter_planning(env,agent):
    if env.action_mode == 'policy':
        return 2, None
   
    if agent.position[0] not in [1,env.dim[0]-1] and agent.position[1] not in [1,env.dim[1]-1]:
        if agent.direction == 0:
            return env.get_action('EAST'),agent.target   
        
        if agent.direction == np.pi/2:
                return env.get_action('NORTH'),agent.target   
        
        if agent.direction == np.pi:
            return env.get_action('WEST'), agent.target
        
        if agent.direction == 3*np.pi/2:
            return env.get_action('SOUTH'), agent.target

    else:
        if agent.position[0]==1 and agent.position[1]!=1:
            return env.get_action('SOUTH'),agent.target
        
        if agent.position[1]==1 and agent.position[0]!=env.dim[0]-1:
            return env.get_action('EAST'),agent.target
        
        if agent.position[0]==env.dim[0]-1 and agent.position[1]!=env.dim[1]-1:
            return env.get_action('NORTH'),agent.target
        
        return env.get_action('WEST'),agent.target