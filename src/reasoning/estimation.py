import random as rd
from copy import *

from src.reasoning.OEATA import HistoryElement
from src.reasoning.pomcp_estimation import *

def process_oeata(unknown_agent, current_state, just_finished_tasks):
    # 1. Initialising the parameter variables
    po = False
    cts_agent = unknown_agent.copy()

    # 'Start process OEATA'
    for set_of_estimators in unknown_agent.smart_parameters['estimations'].learning_data.all_estimators:
        if unknown_agent.smart_parameters['last_completed_task'] != None:
            if unknown_agent != current_state.get_adhoc_agent():
                unknown_agent.smart_parameters['estimations'].learning_data.evaluation(set_of_estimators, cts_agent,
                                                                                       current_state)
                unknown_agent.smart_parameters['estimations'].learning_data.generation(set_of_estimators, cts_agent,
                                                                                       current_state)
        elif just_finished_tasks:
            unknown_agent.smart_parameters['estimations'].learning_data.update_estimators(set_of_estimators,
                                                                                            cts_agent,
                                                                                            current_state,
                                                                                            just_finished_tasks)

        new_estimated_parameter, type_probability = unknown_agent.smart_parameters[
            'estimations'].learning_data.estimation(set_of_estimators)

        # todo: fix it
        unknown_agent.smart_parameters['estimated_parameter'] = new_estimated_parameter
        for estimation_history in unknown_agent.smart_parameters['estimations'].estimation_histories:
            if set_of_estimators.type == estimation_history.type:
                #print ('new_estimated_parameter')
                #print(new_estimated_parameter)
                if new_estimated_parameter is None:
                    estimation_history.estimation_history.append(estimation_history.estimation_history[-1])
                else:
                    estimation_history.estimation_history.append(new_estimated_parameter)
                estimation_history.type_probability = type_probability

<<<<<<< HEAD
    # 'End of Process'
    # te.type_probability = pf_type_probability
    #
    #     # d. If a load action was performed, restart the estimation process
    # todo: change unknown_agent to cts_agent. What is their differences
    # if unknown_agent.next_action == 4 and unknown_agent.is_item_nearby(current_state.items) != -1:
=======
    'End of Process'

>>>>>>> 948dac725416a8a010b6ed5acdfed03f473d721c
    if unknown_agent.smart_parameters['last_completed_task'] != None:
        if unknown_agent.smart_parameters['choose_task_state'] != None:
            hist = HistoryElement(unknown_agent.smart_parameters['choose_task_state'].copy()) \

            unknown_agent.smart_parameters['estimations'].learning_data.history_of_tasks.append(hist)

        unknown_agent.smart_parameters['choose_task_state'] = current_state.copy()
<<<<<<< HEAD
        # unknown_agent.choose_target_pos = unknown_agent.get_position()
        # unknown_agent.choose_target_direction = unknown_agent.direction
    unknown_agent.smart_parameters['estimations'].normalize_type_probabilities()
    return unknown_agent
=======

    normalize_type_probabilities(unknown_agent.smart_parameters['estimations'])
>>>>>>> 948dac725416a8a010b6ed5acdfed03f473d721c


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


def normalize_type_probabilities(estimations):
    # 1. Defining the values
    # print 'Normalizing:',self.l1_estimation.type_probability , self.l2_estimation.type_probability,self.l3_estimation.type_probability, self.l4_estimation.type_probability
    sum_of_probabilities = 0
    type_belief_values = []
    for te in estimations.estimation_histories:
        type_belief_values.append(te.type_probability)
        sum_of_probabilities += te.type_probability

    # 3. Normalising
    if sum_of_probabilities != 0:
        belief_factor = 1 / float(sum_of_probabilities)
        for te in estimations.estimation_histories:
            te.type_probability *= belief_factor
            te.type_probabilities.append(te.type_probability)

    else:
        probabilities = estimations.generate_equal_probabilities()
        for i in range(len(estimations.estimation_histories)):
            estimations.estimation_histories[i].type_probability = probabilities[i]


def level_foraging_uniform_estimation(env, just_finished_tasks):
    adhoc_agent = env.get_adhoc_agent()
    tmp_env = env.copy()
    if env.visibility == 'partial':

        for agent in env.components['agents']:
            if agent != env.get_adhoc_agent() and \
                    None in [agent.level, agent.radius, agent.angle, agent.type]:
                process_oeata(agent, tmp_env, just_finished_tasks)
                agent.level = rd.uniform(0, 1)
                agent.radius = rd.uniform(0, 1)
                agent.angle = rd.uniform(0, 1)
                # Removed 'l4' for now
                agent.type = rd.sample(['l1', 'l2', 'l3'], 1)[0]

        for task in env.components['tasks']:
            if task.level is None:
                task.level = rd.uniform(0, 1)

    else:
        for agent in env.components['agents']:
            if agent != env.get_adhoc_agent():
                process_oeata(agent, tmp_env, just_finished_tasks)

    return env


def capture_uniform_estimation(env, just_finished_tasks):
    adhoc_agent = env.get_adhoc_agent()
    tmp_env = env.copy()
    if env.visibility == 'partial':

        for agent in env.components['agents']:
            if agent != env.get_adhoc_agent() and \
                    None in [agent.level, agent.radius, agent.angle, agent.type]:
                process_oeata(agent, tmp_env, just_finished_tasks)
                agent.level = rd.uniform(0, 1)
                agent.radius = rd.uniform(0, 1)
                agent.angle = rd.uniform(0, 1)
                # Removed 'l4' for now
                agent.type = rd.sample(['c1', 'c2', 'c3'], 1)[0]

        for task in env.components['tasks']:
            if task.level is None:
                task.level = rd.uniform(0, 1)

    else:
        for agent in env.components['agents']:
            if agent != env.get_adhoc_agent():
                agent = process_oeata(agent, env, just_finished_tasks) 

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