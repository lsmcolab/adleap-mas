from gym import spaces
import matplotlib.pyplot as plt

import sys
sys.path.append('src/reasoning')
from scenario_generator import *
from src.reasoning.estimation import *
from src.envs.CaptureEnv import CaptureEnv, Agent, Task
from src.log import LogFile
from src.reasoning.estimation import aga_estimation, abu_estimation, oeata_estimation, pomcp_estimation

###
# Main
###
# TODO : fix pomcp black box .
components = {
    'agents':[
        Agent(index='A',atype="pomcp",position=(1,0),direction=3*np.pi/2,radius=1,angle=1),
        Agent(index='B',atype='c1',position=(8,2),direction=3*np.pi/2,radius=0.40,angle=0.19),
       Agent(index='C',atype='c1',position=(6,4),direction=3*np.pi/2,radius=0.15,angle=0.4),
        Agent(index='D',atype='c1',position=(2,7),direction=np.pi,radius=0.11,angle=0.41),
    ],
    'adhoc_agent_index': 'A',
    'tasks':[Task('1',(7,0)),
            Task('2',(8,1)),
            Task('3',(7,4)),
            Task('4',(5,6))]}

env = CaptureEnv((10,10),components,visibility='full',display=True)
# env = LevelForagingEnv((10,10),components,visibility='full')
env.agents_color = {'l1':'lightgrey','l2':'darkred','l3':'darkgreen','l4':'darkblue',\
                        'entropy':'blue','mcts':'yellow','pomcp':'red'}
state = env.reset()
#log_file = LogFile(env)


#################################################


estimation = "OEATA"
rounds = 1
loss = []
time_step = 0
for i in range(rounds):
    state = env.reset()
    done = False
    adhoc_agent = env.get_adhoc_agent()

    if estimation == 'OEATA':
        adhoc_agent.smart_parameters['estimation_args'] =\
        get_env_types("CaptureEnv"), [(0,1),(0,1)]
        estimation_method = aga_estimation


    while not done and env.episode < 15:
        # Rendering the environment

        #env.render()
        print("Episode : ", env.episode)
        # Main Agent taking an action
        module = __import__(adhoc_agent.type)
        method = getattr(module, adhoc_agent.type+'_planning')

        if(adhoc_agent.type == "mcts" or adhoc_agent.type=="pomcp"):
            adhoc_agent.next_action, adhoc_agent.target = method(state,adhoc_agent,estimation_algorithm=estimation_method)
        else:
            adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)



        # Step on environment
        state, reward, done, info = env.step(adhoc_agent.next_action)
        just_finished_tasks = info['just_finished_tasks']


        # Verifying the end condition

        print(adhoc_agent.smart_parameters['estimation'].get_estimation(env))


        if done:
            break





    env.close()