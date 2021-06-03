from gym import spaces
import matplotlib.pyplot as plt
from src.reasoning.fundamentals import FundamentalValues
from src.reasoning.OEATA import *
from src.reasoning.parameter_estimation import *

import sys
sys.path.append('src/reasoning')

from src.reasoning.estimation import *
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
from src.log import LogFile

from src.reasoning.AGA import *
from src.reasoning.ABU import *

###
# Main
###
components = {
    'agents':[
        Agent(index='A',atype='l1',position=(0,0),direction=np.pi/2,radius=0.1,angle=0.1,level=0.1),
        Agent(index='B',atype='l1',position=(0,9),direction=np.pi/2,radius=0.7,angle=0.5,level=1.0),
        Agent(index='C',atype='l2',position=(9,9),direction=np.pi/2,radius=1.0,angle=1.0,level=1.0),
        Agent(index='D',atype='l3',position=(9,0),direction=np.pi/2,radius=0.6,angle=0.7,level=1.0),
    ],
    'adhoc_agent_index': 'A',
    'tasks':[Task('1',(2,2),1.0),
            Task('2',(4,4),1.0),
            Task('3',(5,5),1.0),
            Task('4',(8,8),1.0)]}


#####################################################

env = LevelForagingEnv((10,10),components,visibility='full')
env.agents_color = {'l1':'lightgrey','l2':'darkred','l3':'darkgreen','l4':'darkblue',\
                        'entropy':'blue','mcts':'yellow','pomcp':'red'}
state = env.reset()
#log_file = LogFile(env)

rounds = 1

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
    aga = AGAprocess(estimation_config, env)

elif estimation_mode == "ABU":
    estimation_config = ABUConfig(fundamental_values, 4)
    abu = ABUprocess(estimation_config, env)



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

        # Main Agent taking an action
        module = __import__(adhoc_agent.type)
        method = getattr(module, adhoc_agent.type+'_planning')
        if(adhoc_agent.type == "mcts" or adhoc_agent.type=="pomcp"):
            adhoc_agent.next_action, adhoc_agent.target = method(state,adhoc_agent)
        else:
            adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)

        if(estimation_mode=='AGA'):
            aga.update(env)
        elif(estimation_mode == 'ABU'):
            abu.update(env)
        # Step on environment
        state, reward, done, info = env.step(adhoc_agent.next_action)
        just_finished_tasks = info['just_finished_tasks']

        # just_finished_tasks.append(env.components['tasks'][1])
        # print (done)

        print (len(just_finished_tasks))
        #log_file.write(env)

        if(estimation_mode == 'OEATA'):
            level_foraging_uniform_estimation(env, just_finished_tasks)

        print(env.components['tasks'][1].completed)
        print(env.components['tasks'][2].completed)
        print(env.components['tasks'][3].completed)
        print(env.components['tasks'][0].completed)
        print ("---------------------------------")
        # Verifying the end condition
        if done:
            break

        for agent in env.components['agents']:
            if(agent.index != adhoc_agent.index):
                selected_type = agent.smart_parameters['estimations'].get_highest_type_probability()
                selected_parameter = agent.smart_parameters['estimations'].get_parameters_for_selected_type(selected_type)
                print(agent.index,selected_type,selected_parameter.radius,selected_parameter.angle,selected_parameter.level)

    env.close()
