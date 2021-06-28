import random as rd
import numpy as np
from copy import *

from src.reasoning.pomcp_estimation import *

def aga_estimation(env, adhoc_agent,\
 template_types, parameters_minmax, grid_size=4, reward_factor=0.04, step_size=0.01, decay_step=0.999, degree=2, univariate=True):
    #####
    # AGA INITIALISATION
    #####
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from aga import AGA
        adhoc_agent.smart_parameters['estimation'] = AGA(env,template_types,parameters_minmax,grid_size,\
                                                        reward_factor,step_size,decay_step,degree,univariate)
        
    #####    
    # AGA PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.smart_parameters['estimation'].update(env)

    #####
    # AGA ESTIMATION
    #####
    return env, adhoc_agent.smart_parameters['estimation']

def abu_estimation(env, adhoc_agent, \
 template_types, parameters_minmax, grid_size=4, reward_factor=0.04, degree=2):
    #####
    # ABU INITIALISATION
    #####
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from abu import ABU
        adhoc_agent.smart_parameters['estimation'] = ABU(env,template_types,parameters_minmax,grid_size,\
                                                        reward_factor,degree)
        
    #####    
    # ABU PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.smart_parameters['estimation'].update(env)

    #####
    # ABU ESTIMATION
    #####
    return env, adhoc_agent.smart_parameters['estimation']

def oeata_estimation(env, adhoc_agent,\
 template_types, nparameters, N=100, xi=2, mr=0.2, d=100, normalise=np.mean):
    #####
    # OEATA INITIALISATION
    #####
    # Initialising the oeata method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from oeata import OEATA
        adhoc_agent.smart_parameters['estimation'] = OEATA(env,template_types,nparameters,N,xi,mr,d,normalise)
        
    #####    
    # OEATA PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.smart_parameters['estimation'].run(env)

    #####
    # OEATA - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def process_pomcp_estimation(env):
    iteration_max = 100
    max_depth = 100
    particle_filter_numbers = 100

    for unknown_agent in env.components['agents']:
        if unknown_agent != env.get_adhoc_agent():

            pomcpe = POMCP(iteration_max, max_depth, particle_filter_numbers)
            estimated_parameter, estimated_type = pomcpe.start_estimation(None, env)

            for estimation_history in unknown_agent.smart_parameters['estimations'].estimation_histories:
                if estimated_parameter is None:
                    estimation_history.estimation_history.append(estimation_history.estimation_history[-1])
                else:
                    estimation_history.estimation_history.append(estimated_parameter)
                estimation_history.type_probability = 1

def level_foraging_uniform_estimation(env, template_types=['l1','l2','l3']):
    adhoc_agent = env.get_adhoc_agent()
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            teammate.type = rd.sample(template_types,1)[0]
            teammate.set_parameters(np.random.uniform(0,1,3))
    return env

def capture_uniform_estimation(env):
    raise NotImplemented

def truco_uniform_estimation(env):
    if env.visibility == 'partial':
        for player in env.components['player']:
            if player != env.components['player'][env.current_player]:
                player.hand = []
                while len(player.hand) < 3:
                    player.hand.append(env.components['cards in game'].pop(0))
                player.type = rd.sample(['t1', 't2', 't3'], 1)[0]

    env.visibility = 'full'
    return env
