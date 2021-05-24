from gym import spaces
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('src/reasoning')

from src.reasoning.estimation import level_foraging_uniform_estimation
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.envs.CaptureEnv import CaptureEnv,Agent,Task
from src.log import LogFile
from src.reasoning.AGA import *
###
# Main
###
# TODO : fix pomcp black box .
components = {
    'agents':[
        Agent(index='A',atype="l3",position=(4,5),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
        Agent(index='B',atype='l3',position=(5,4),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
       Agent(index='C',atype='l3',position=(6,5),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
        Agent(index='D',atype='l3',position=(5,6),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
    ],
    'adhoc_agent_index': 'A',
    'tasks':[Task('1',(0,0),1.0),
            Task('2',(9,9),1.0),
            Task('3',(5,5),1.0),
            Task('4',(8,8),1.0)]}

env = CaptureEnv((10,10),components,visibility='partial')
# env = LevelForagingEnv((10,10),components,visibility='full')
env.agents_color = {'l1':'lightgrey','l2':'darkred','l3':'darkgreen','l4':'darkblue',\
                        'entropy':'blue','mcts':'yellow','pomcp':'red'}
state = env.reset()
log_file = LogFile(env)

rounds = 1
epsilon = 0.80
decay = 0.99
step_size = 0.01
loss = []
time_step = 0
for i in range(rounds):
    state = env.reset()
    done = False
    adhoc_agent = env.get_adhoc_agent()
    while not done and env.episode < 30:
        # Rendering the environment
        env.render()
        #AGA(state, adhoc_agent,epsilon,step_size)

        # Uncomment for extra information

        #print_stats(adhoc_agent)
        #l = AGA_loss(env,adhoc_agent)
        #if (not l is None and time_step%100==0):
        #    loss.append(l)

        #time_step+=1
        # Main Agent taking an action
        module = __import__(adhoc_agent.type)
        method = getattr(module, adhoc_agent.type+'_planning')
        if(adhoc_agent.type == "mcts" or adhoc_agent.type=="pomcp"):
            adhoc_agent.next_action, adhoc_agent.target = method(state,adhoc_agent,
                                            estimation_algorithm=level_foraging_uniform_estimation)
        else:
            adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)

        # Step on environment
        state, reward, done, info = env.step(adhoc_agent.next_action)
        log_file.write(env)
        # Verifying the end condition
        if done:
            break
    epsilon = epsilon*decay
    step_size = step_size*decay
    env.close()