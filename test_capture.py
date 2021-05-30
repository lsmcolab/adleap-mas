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

env = CaptureEnv((10,10),components,visibility='full')
# env = LevelForagingEnv((10,10),components,visibility='full')
env.agents_color = {'l1':'lightgrey','l2':'darkred','l3':'darkgreen','l4':'darkblue',\
                        'entropy':'blue','mcts':'yellow','pomcp':'red'}
state = env.reset()
log_file = LogFile(env)

# Estimator Configuration
estimation_mode = 'ABU'
oeata_parameter_calculation_mode = 'MEAN'   #  It can be MEAN, MODE, MEDIAN
agent_types = ['l1','l2','l3']
estimators_length = 100
mutation_rate = 0.2
aga,abu = None,None

fundamental_values = FundamentalValues(radius_max=1, radius_min=0.1, angle_max=1, angle_min=0.1, level_max=1,
                                       level_min=0, agent_types=agent_types, env_dim=[10, 10],
                                       estimation_mode=estimation_mode)


if estimation_mode == 'AGA':
    estimation_config = AGAConfig(fundamental_values, 4, 0.01, 0.999)
    aga = AGAprocess(estimation_config, state)

elif estimation_mode == "ABU":
    estimation_config = ABUConfig(fundamental_values, 4)
    abu = ABUprocess(estimation_config, state)

#################################################



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


    while not done and env.episode < 30:
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
            aga.update(env)
        elif (estimation_mode == 'ABU'):
            abu.update(env)

        # Step on environment
        state, reward, done, info = env.step(adhoc_agent.next_action)
        log_file.write(env)
        # Verifying the end condition
        if done:
            break
    epsilon = epsilon*decay
    step_size = step_size*decay

#    if (estimation_mode == 'OEATA'):
#        level_foraging_uniform_estimation(env, just_finished_tasks)

    for agent in env.components['agents']:
        if(agent.index != adhoc_agent.index):
            print(agent.smart_parameters['estimations'].estimation_histories[0].get_estimation_history())

    env.close()