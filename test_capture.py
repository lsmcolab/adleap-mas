from gym import spaces
import matplotlib.pyplot as plt
from src.reasoning.fundamentals import FundamentalValues
from src.reasoning.OEATA import *
from src.reasoning.parameter_estimation import *

import sys
sys.path.append('src/reasoning')

from src.reasoning.estimation import *
from src.envs.CaptureEnv import CaptureEnv, Agent, Task
from src.log import LogFile

from src.reasoning.AGA import *
from src.reasoning.ABU import *


###
# Main
###
# TODO : fix pomcp black box .
components = {
    'agents':[
        Agent(index='A',atype="mcts",position=(1,0),direction=3*np.pi/2,radius=0.6,angle=0.16,level=0.9),
        Agent(index='B',atype='c1',position=(8,2),direction=3*np.pi/2,radius=0.40,angle=0.19,level=0.58),
       Agent(index='C',atype='c3',position=(6,4),direction=3*np.pi/2,radius=0.15,angle=0.4,level=0.7),
        Agent(index='D',atype='c1',position=(2,7),direction=np.pi,radius=0.11,angle=0.41,level=0.57),
    ],
    'adhoc_agent_index': 'A',
    'tasks':[Task('1',(7,0),1.0),
            Task('2',(8,1),1.0),
            Task('3',(7,4),1.0),
            Task('4',(5,6),1.0)]}

env = CaptureEnv((10,10),components,visibility='full',display=False)
# env = LevelForagingEnv((10,10),components,visibility='full')
env.agents_color = {'l1':'lightgrey','l2':'darkred','l3':'darkgreen','l4':'darkblue',\
                        'entropy':'blue','mcts':'yellow','pomcp':'red'}
state = env.reset()
#log_file = LogFile(env)

# Estimator Configuration
estimation_mode = 'AGA'
oeata_parameter_calculation_mode = 'MEAN'   #  It can be MEAN, MODE, MEDIAN
agent_types = ['c1','c2','c3']
estimators_length = 100
mutation_rate = 0.2
aga,abu = None,None

fundamental_values = FundamentalValues(radius_max=1, radius_min=0.1, angle_max=1, angle_min=0.1, level_max=1,
                                       level_min=0, agent_types=agent_types, env_dim=[10, 10],
                                       estimation_mode=estimation_mode)


if estimation_mode == 'AGA':
    estimation_config = AGAConfig(fundamental_values, 4, 0.01, 0.999)
    aga = AGAprocess(estimation_config, env)

elif estimation_mode == "ABU":
    estimation_config = ABUConfig(fundamental_values, 4)
    abu = ABUprocess(estimation_config, env)

#################################################



rounds = 1
loss = []
time_step = 0
for i in range(rounds):
    state = env.reset()
    done = False
    adhoc_agent = env.get_adhoc_agent()


    if estimation_mode == 'OEATA':
        estimation_config = OeataConfig(estimators_length, oeata_parameter_calculation_mode, mutation_rate,
                                                 fundamental_values)

        for a in env.components['agents']:
            a.smart_parameters['last_completed_task'] = None
            a.smart_parameters['choose_task_state'] = env.copy()
            if a.index != adhoc_agent.index:
                param_estim = ParameterEstimation(estimation_config)
                param_estim.estimation_initialisation()
                oeata = OEATA_process(estimation_config, a)
                oeata.initialisation(a.position, a.direction, a.radius,a.angle,a.level, env)
                param_estim.learning_data = oeata
                a.smart_parameters['estimations'] = param_estim


    while not done and env.episode < 10:
        # Rendering the environment

        env.render()
        print("Episode : ", env.episode)
        # Main Agent taking an action
        module = __import__(adhoc_agent.type)
        method = getattr(module, adhoc_agent.type+'_planning')

        if(adhoc_agent.type == "mcts" or adhoc_agent.type=="pomcp"):
            adhoc_agent.next_action, adhoc_agent.target = method(state,adhoc_agent)
        else:
            adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)

        if (estimation_mode == 'AGA'):
            aga.update(state)
        elif (estimation_mode == 'ABU'):
            abu.update(state)


        # Step on environment
        state, reward, done, info = env.step(adhoc_agent.next_action)
        just_finished_tasks = info['just_finished_tasks']


        # Verifying the end condition

        if (estimation_mode == 'OEATA'):
            capture_uniform_estimation(env, just_finished_tasks)

        for agent in env.components['agents']:
            print(agent.index)

        print(env.components['tasks'][1].completed,len(just_finished_tasks))
        print(env.components['tasks'][2].completed)
        print(env.components['tasks'][3].completed)
        print(env.components['tasks'][0].completed)
        print("---------------------------------")

        for agent in env.components['agents']:
            if(agent.index != adhoc_agent.index):
                selected_type = agent.smart_parameters['estimations'].get_highest_type_probability()
                selected_parameter = agent.smart_parameters['estimations'].get_parameters_for_selected_type(selected_type)
                print(agent.index,selected_type,selected_parameter.radius,selected_parameter.angle,selected_parameter.level)



        if done:
            break





    env.close()