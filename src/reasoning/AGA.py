import numpy as np
import sklearn.linear_model
from sklearn.metrics import log_loss
import random

import warnings

# Uncomment below to remove warnings
#warnings.filterwarnings("ignore")

from src.reasoning import *

def choose_types(type_prob,epsilon=0.1):
    if(random.uniform(0,1)<epsilon):
        return random.randint(0,len(type_prob)-1)
    else:
        return np.argmax(type_prob)


def AGA(env,agent,epsilon=0.1,step=0.01):
    """ AGA estimation method
    :param env: Observation state of the agent.
    :param agent: The adhoc reasoning agent
    :param epsilon: Exploration vs Exploitation trade off
    :param step: Update step size

    agent.smart_parameters['agents_aga'] stores a prediction of each agent once it sees it. This prediction is updated
    at each call . Each agent is identified by it's index.
    """
    # Storing a copy of the observation for next step
    step_size = step
    n = 2
    types = ["l1","l2","l3","l4"]
    # Initializing AGA predictions
    if('agents_aga' not in agent.smart_parameters.keys()):
        agent.smart_parameters['agents_aga'] = {}

        agent.smart_parameters['prev_episode'] = env.copy()
        return
    if(env.episode == 1):
        agent.smart_parameters['prev_episode'] = env.copy()
        return
    #
    for agents in env.components['agents']:
        if(agents.index not in agent.smart_parameters["agents_aga"].keys()):
            agent.smart_parameters["agents_aga"][agents.index] = [agents.copy(),[1/len(types) for type in types]]
            agent.smart_parameters["agents_aga"][agents.index][0].radius = random.uniform(0,1)
            agent.smart_parameters["agents_aga"][agents.index][0].angle = random.uniform(0, 1)
            agent.smart_parameters["agents_aga"][agents.index][0].level = random.uniform(0, 1)

    # Grid of parameters
    grid = []
    for i in np.linspace(0,1,n):
        for j in np.linspace(0,1,n):
            for k in np.linspace(0,1,n):
                grid.append([i,j,k])


    for agents in env.components['agents']:
        pred_action = []
        index = agents.index
        type_index = 0
        actual_action = agents.next_action
        pos = None

        old_env = agent.smart_parameters["prev_episode"]
        #Meaningful updates occur only if an agent is visible in current and previous state.
        for i in range(0,len(old_env.components["agents"])):
            if(old_env.components["agents"][i].index == index):
                pos = i

        if(pos == None):
            continue
        for [radius,angle,level] in grid:
            env_copy = old_env.copy()
            for copy_agents in env_copy.components['agents']:
                if(copy_agents.index==index):
                    copy_agents.radius = radius
                    copy_agents.angle = angle
                    copy_agents.level = level
                    copy_agents.target = None
                    type_index = choose_types(agent.smart_parameters["agents_aga"][index][1],epsilon)
                    copy_agents.type = types[type_index]
                else:
                    copy_agents.radius = random.uniform(0,1)
                    copy_agents.angle = random.uniform(0,1)
                    copy_agents.level = random.uniform(0,1)
                    copy_agents.target=None
                    copy_agents.type = random.sample(types, 1)[0]

            # Can use any step
            env_copy.step(0)
            pred_action.append(env_copy.components['agents'][pos].next_action)
            if(env_copy.components["agents"][pos].next_action==actual_action):
                agent.smart_parameters["agents_aga"][index][1][type_index]*=0.905
            else:
                agent.smart_parameters["agents_aga"][index][1][type_index] *= 1.095

            for t in range(0,len(types)):
                agent.smart_parameters["agents_aga"][index][1][t] /= sum(agent.smart_parameters['agents_aga'][index][1])
        y = [0.96 if (pred_action[i] == actual_action) else 0.01 for i in range(0,len(pred_action))]
        model = sklearn.linear_model.LinearRegression()
        model.fit(np.array(grid),y)

        # Updating the predictions
        agent.smart_parameters['agents_aga'][index][0].radius += step_size * model.coef_[0]
        agent.smart_parameters['agents_aga'][index][0].angle += step_size * model.coef_[1]
        agent.smart_parameters['agents_aga'][index][0].level += step_size * model.coef_[2]

    agent.smart_parameters["prev_episode"] = env.copy()

    return

def print_stats(adhoc_agent):
    if("agents_aga" not in adhoc_agent.smart_parameters.keys()):
        return
    stats = {}
    for i in adhoc_agent.smart_parameters["agents_aga"].keys():
        if(i==adhoc_agent.index):
            continue
        agent = adhoc_agent.smart_parameters["agents_aga"][i][0]
        types = adhoc_agent.smart_parameters["agents_aga"][i][1]
        stats[agent.index] = " Radius : " + str(agent.radius) + " Angle : " + str(agent.angle) + " Level : " + str(agent.level) \
                                        + " Type : " + str(types) + " "

    print(stats)

# To calculate performance of AGA
def AGA_loss(env,adhoc_agent):
    if ("agents_aga" not in adhoc_agent.smart_parameters.keys()):
        return -1

    loss = {'radius':0,'angle':0,'level':0,'type':0}
    total_agent = 0
    one_hot = {"l1" : [1,0,0,0], "l2":[0,1,0,0], "l3":[0,0,1,0], "l4" : [0,0,0,1]}
    for agent in env.components["agents"]:
        if (agent.index == adhoc_agent.index or agent.index not in adhoc_agent.smart_parameters["agents_aga"].keys()):
            continue
        total_agent += 1
        agent_pred = adhoc_agent.smart_parameters["agents_aga"][agent.index][0]
        types = adhoc_agent.smart_parameters["agents_aga"][agent.index][1]
        loss["radius"] += (agent.radius - agent_pred.radius)**2
        loss["angle"] += (agent.angle - agent_pred.angle) ** 2
        loss["level"] += (agent.level - agent_pred.level)**2
        loss["type"] += log_loss(one_hot[agent.type],np.array(types))

    if(total_agent == 0):
        return

    for key in ["radius","angle","level"]:
        loss[key] = np.sqrt(loss[key]/total_agent)
    loss["type"] = loss["type"]/total_agent

    return loss
