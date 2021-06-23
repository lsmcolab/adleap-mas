import random as rd
import numpy as np
from copy import *

from src.reasoning.pomcp_estimation import *

def oeata_estimation(env, adhoc_agent, template_types, nparameters, N=100, xi=2, mr=0.2, d=100, normalise=np.mean):
    #####
    # OEATA INITIALISATION
    #####
    # Initialising the oeata method inside the adhoc agent
    if 'oeata' not in adhoc_agent.smart_parameters:
        from oeata import OEATA
        adhoc_agent.smart_parameters['oeata'] = OEATA(env,template_types,nparameters,N,xi,mr,d,normalise)
        
    #####    
    # OEATA PROCESS
    #####
    adhoc_agent.smart_parameters['oeata'] = adhoc_agent.smart_parameters['oeata'].run(env)

    #####
    # OEATA ESTIMATION
    #####
    types, _, param_est =\
        adhoc_agent.smart_parameters['oeata'].get_estimation(env)

    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            teammate.type = types[teammate.index]
            teammate.set_parameters(param_est[teammate.index])

    return env, adhoc_agent.smart_parameters['oeata']

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

def level_foraging_uniform_estimation(env, just_finished_tasks):
    raise NotImplemented

def capture_uniform_estimation(env, just_finished_tasks):
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