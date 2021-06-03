from a_star import a_star_planning
import numpy as np
import random

# Chooses the closest task . Same as l1, but works with task index instead
def c3_planning(env,agent):
    # action = None
    # if(agent.target):
    #     for t in env.components['tasks']:
    #         if(agent.target == t.index):
    #             if((agent.position[0]-t.position[0])**2 + (agent.position[1]-t.position[1])**2 == 1):
    #                 if(random.uniform(0,1)<0.5):
    #                     action = 4
    #                     return action,agent.target
    #                 else:
    #                     steps =  [(-1,0),(0,-1),(1,0),(0,1)]
    #                     poss = []
    #                     for i,step in enumerate(steps):
    #                         (w,h) = env.state.shape
    #                         if(0<=t.position[0]+step[0]<w and 0<=t.position[1]+step[1]<h):
    #                             new_pos = (t.position[0]+step[0],t.position[1]+step[1])
    #                             if( env.state[new_pos[0],new_pos[1]] == 0):
    #                                 poss.append((i,new_pos))
    #                     if(len(poss)==0):
    #                         return 4,agent.target
    #                     action = random.sample(poss,1)[0]
    #                     agent.target_position = action[1]
    #             else:
    #                 agent.target_position = t.position
    #
    # else:
    #     agent.target = choose_target(env,agent)
    #
    # if(agent.target == None):
    #     action = random.sample([0,1,2,3,4],1)[0]
    #     return action,None
    # next_action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
    #                               env.action_space, agent.position, agent.target_position)
    #
    action = None
    agent_task = None
    task_position = None
    if(agent.target):
        for t in env.components['tasks']:
            if(t.index == agent.target and not t.completed):
                agent_task = t
                task_position = t.position

    if(agent_task is not None):
        agent.target_position = task_position
        action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                   env.action_space, agent.position, agent.target_position)
        return action,agent.target
    else:
        agent.target = choose_target(env,agent)
        if(agent.target is None):
            return random.sample([0,1,2,3,4],1)[0],None

        for t in env.components['tasks']:
            if(t.index == agent.target):
                agent.target_position = t.position

        action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                 env.action_space, agent.position, agent.target_position)
        return action, agent.target




def choose_target(env, agent):
    state = env.state
    # 0. Initialising the support variables
    nearest_task_idx, min_distance = -1, np.inf

    # 1. Searching for max distance item
    visible_tasks = [(x,y) for x in range(state.shape[0])
                        for y in range(state.shape[1]) if state[x,y] == np.inf]

    for i in range(0, len(visible_tasks)):
        dist = distance(visible_tasks[i],agent.position)
        if dist < min_distance:
            min_distance = dist
            nearest_task_idx = i

    # 2. Verifying the found task
    # a. no task found
    if nearest_task_idx == -1:
        return None
    # b. task found
    else:
        agent.target_position = visible_tasks[nearest_task_idx]
        task_idx = None
        for t in env.components['tasks']:
            if(t.position == agent.target_position):
                task_idx = t.index

        return task_idx

def distance(obj, viewer):
    return np.sqrt((obj[0] - viewer[0])**2 + (obj[1] - viewer[1])**2)
