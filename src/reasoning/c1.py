from a_star import a_star_planning
import numpy as np
import random

# Pursues odd index preys
def c1_planning(env,agent):
    action, target_pos = None, None
    
    # checking if the agent already chosen a prey
    if(agent.target):
        for t in env.components['tasks']:
            if(t.index == agent.target ):
                # if it is completed, pursue a new one
                if(t.completed):
                    agent.target = choose_target(env,agent)
                    target_pos = agent.target_position
                # else, keep going
                else:
                    target_pos = t.position

        # defining the path to the prey
        if(target_pos):
            agent.target_position = target_pos
            action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                     env.action_space, agent.position, agent.target_position)
            return action, agent.target
        else:
            action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                     env.action_space, agent.position, agent.target_position)
            return action, agent.target

    # else, choose one
    else:
        agent.target = choose_target(env,agent)

        # if did not find a valid target, move randomly
        if(not agent.target):
            action = env.action_space.sample()
            return action, None
        # else pursue it
        else:
            action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                   env.action_space, agent.position, agent.target_position)

            return action,agent.target

def choose_target(env, agent):
    for task in env.components['tasks']:
        if int(task.index) % 2 == 1:
            agent.target_position = task.position
            return task.index
    return None
