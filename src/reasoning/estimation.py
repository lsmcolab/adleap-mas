import random as rd
import numpy as np
from copy import *
from src.envs.SmartFireBrigadeEnv import SmartFireBrigadeEnv
from src.envs.CaptureEnv import CaptureEnv
from src.envs.LevelForagingEnv import LevelForagingEnv
from src.reasoning.pomcp_estimation import *

def aga_estimation(env, adhoc_agent,\
 template_types, parameters_minmax, grid_size=100, reward_factor=0.04, step_size=0.01, decay_step=0.999, degree=2, univariate=True):
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
    # AGA - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def abu_estimation(env, adhoc_agent, \
 template_types, parameters_minmax, grid_size=100, reward_factor=0.04, degree=2):
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
    # ABU - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def oeate_estimation(env, adhoc_agent,\
 template_types, parameters_minmax, N=100, xi=2, mr=0.2, d=100, normalise=np.mean):
    #####
    # OEATE INITIALISATION
    #####
    # Initialising the oeata method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from oeate import OEATE
        adhoc_agent.smart_parameters['estimation'] = OEATE(env,template_types,parameters_minmax,\
                                                                                N,xi,mr,d,normalise)
        
    #####    
    # OEATE PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.smart_parameters['estimation'].run(env)

    #####
    # OEATE - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)

    return env, adhoc_agent.smart_parameters['estimation']

def pomcp_estimation(env, adhoc_agent, \
 template_types, parameters_minmax, discount_factor=0.9, max_iter=100, max_depth=10,min_particles=100):
    #####
    # POMCE INITIALISATION
    ##### discount_factor=0.9,max_iter=10,max_depth=10,min_particles=100
    # Initialising the aga method inside the adhoc agent
    if 'estimation' not in adhoc_agent.smart_parameters:
        from pomce import POMCE
        adhoc_agent.smart_parameters['estimation'] = POMCE(env,template_types,parameters_minmax,discount_factor,max_iter,max_depth,min_particles)
        
    #####    
    # POMCE PROCESS
    #####
    adhoc_agent.smart_parameters['estimation'] = adhoc_agent.smart_parameters['estimation'].update(env)

    #####
    # POMCP - SET ESTIMATION
    #####
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            selected_type = adhoc_agent.smart_parameters['estimation'].sample_type_for_agent(teammate)
            selected_parameter = adhoc_agent.smart_parameters['estimation'].get_parameter_for_selected_type(teammate,selected_type)

            teammate.type = selected_type
            teammate.set_parameters(selected_parameter)
  
    return env, adhoc_agent.smart_parameters['estimation']

def uniform_estimation(env):
    if isinstance(env,LevelForagingEnv):
        return level_foraging_uniform_estimation(env)
    elif isinstance(env,CaptureEnv):
        return capture_uniform_estimation(env)
    elif isinstance(env,SmartFireBrigadeEnv):
        return smartfirebrigade_uniform_estimation(env)
    else:
        raise NotImplemented

def level_foraging_uniform_estimation(env, template_types=['l1','l2','l3']):
    adhoc_agent = env.get_adhoc_agent()
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            teammate.type = rd.sample(template_types,1)[0]
            teammate.set_parameters(np.random.uniform(0.5,1,3))
    return env

def capture_uniform_estimation(env, template_types=['c1','c2','c3']):
    adhoc_agent = env.get_adhoc_agent()
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            teammate.type = rd.sample(template_types,1)[0]
            teammate.set_parameters(np.random.uniform(0.5,1,2))
    return env

def smartfirebrigade_uniform_estimation(env):
    # TODO: Implement an uniform estimation for the SFB environment
    return env    

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
