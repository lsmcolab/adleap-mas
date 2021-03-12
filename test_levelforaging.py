from gym import spaces
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('src/reasoning')

from src.reasoning.estimation import level_foraging_uniform_estimation
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.log import LogFile
from src.reasoning.AGA import *
###
# Main
###
components = {
    'agents':[
        Agent(index='A',atype='pomcp',position=(0,0),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
        Agent(index='B',atype='l1',position=(0,9),direction=np.pi/2,radius=0.7,angle=0.5,level=1.0),
       Agent(index='C',atype='l2',position=(9,9),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
        Agent(index='D',atype='l3',position=(9,0),direction=np.pi/2,radius=0.6,angle=0.7,level=1.0),
    ],
    'adhoc_agent_index': 'A',
    'tasks':[Task('1',(2,2),1.0),
            Task('2',(4,4),1.0),
            Task('3',(5,5),1.0),
            Task('4',(8,8),1.0)]}

env = LevelForagingEnv((10,10),components,visibility='full')
env.agents_color = {'l1':'lightgrey','l2':'darkred','l3':'darkgreen','l4':'darkblue',\
                        'entropy':'blue','mcts':'yellow','pomcp':'red'}
state = env.reset()
log_file = LogFile(env)

rounds = 10
epsilon = 0.90
decay = 0.99
step_size = 0.01
loss = []
print(env.copy())
for i in range(rounds):
    state = env.reset()
    done = False
    adhoc_agent = env.get_adhoc_agent()
    while not done and env.episode < 200:
        # Rendering the environment
        #env.render()
        print(env.episode)
        AGA(state, adhoc_agent,epsilon,step_size)

        # Uncomment for extra information

        #print_stats(adhoc_agent)
        #l = AGA_loss(env,adhoc_agent)
        #if (not l is None):
        #    loss.append(l)

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

# plt.plot([x['radius'] for x in loss])
# plt.plot([x['angle'] for x in loss])
# plt.plot([x['level'] for x in loss])
# print([x['radius'] for x in loss])
# print([x['angle'] for x in loss])
# print([x['level'] for x in loss])
# plt.legend(["Radius","Angle","Level"])
# plt.savefig("results/AGA_rad.png")
