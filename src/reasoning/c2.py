from a_star import a_star_planning
import numpy as np
import random

# Chooses the task with maximum sum of coordinates . Same as l2, but works with task index instead
def c2_planning(env,agent):
    action = None
    agent_task = None
    target_pos = None
    if(agent.target):
        for t in env.components['tasks']:
            if(t.index == agent.target ):
                if(t.completed):
                    agent.target = choose_target(env,agent)
                    target_pos = agent.target_position
                else:
                    target_pos = t.position

        if(target_pos):
            agent.target_position = target_pos
            action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                     env.action_space, agent.position, agent.target_position)
            return action, agent.target
        else:
            action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                     env.action_space, agent.position, agent.target_position)
            return action, agent.target


    else:
        agent.target = choose_target(env,agent)
        if(not agent.target):
            action = env.action_space.sample()
            return action, None
        else:
            action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
                                   env.action_space, agent.position, agent.target_position)

            return action,agent.target




def choose_target(env, agent):
    state = env.state
    # 0. Initialising the support variables
    furthest_task_idx, max_distance = -1, -np.inf

    # 1. Searching for max distance item
    visible_tasks = [(x,y) for x in range(state.shape[0])
                        for y in range(state.shape[1]) if state[x,y] == np.inf]

    for i in range(0, len(visible_tasks)):
        dist = distance(visible_tasks[i],agent.position)
        if dist > max_distance:
            max_distance = dist
            furthest_task_idx = i

    # 2. Verifying the found task
    # a. no task found
    if furthest_task_idx == -1:
        return None
    # b. task found
    else:
        agent.target_position = visible_tasks[furthest_task_idx]
        task_idx = None
        for t in env.components['tasks']:
            if(t.position == agent.target_position):
                task_idx = t.index

        return task_idx

def distance(obj, viewer):
    return sum(obj)
