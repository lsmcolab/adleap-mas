import numpy as np
from scipy import stats
import random
from src.envs.LevelForagingEnv import *

from src.reasoning.fundamentals import Parameter


class BagOfEstimators:
    def __init__(self):
        self.radius_bag = []
        self.levels_bag = []
        self.angles_bag = []


class OeataConfig:
    def __init__(self, estimators_length, parameter_calculation_mode, mutation_rate,
                 fundamental_values):
        self.estimators_length = estimators_length
        self.parameter_calculation_mode = parameter_calculation_mode
        self.mutation_rate = mutation_rate
        self.fundamental_values = fundamental_values


class HistoryElement:
    def __init__(self, choose_target_state=None, target=None):
        # , agent_choose_target_position=None,
        #          agent_choose_target_direction=None):
        self.target = target
        self.choose_target_state = choose_target_state
        # self.agent_choose_target_position = agent_choose_target_position
        # self.agent_choose_target_direction = agent_choose_target_direction


class Estimator:
    def __init__(self, parameter=None, choose_target_state=None, target_task=None, success_rate=None, failure_rate=None):
        self.parameter = parameter
        self.choose_target_state = choose_target_state
        self.target_task = target_task
        self.success_rate = success_rate
        self.failure_rate = failure_rate


class SetOfEstimators:
    def __init__(self,  type=None):
        self.Estimators = []
        self.type = type


class OEATA_process:
    def __init__(self, oeata_config, agent=None):
        self.oeata_config = oeata_config
        self.agent = agent
        self.all_estimators = []
        self.history_of_tasks = []
        self.bag_of_estimators = BagOfEstimators()
        self.load_count = 0  # todo: I am not sure if it is really useful

    ################################################################################################################
    def extract_parameter_values(self, set_of_estimators):
        level_values, angle_values, radius_values = [], [], []

        if len(set_of_estimators) == 0:
            return level_values, angle_values, radius_values

        for estimator in set_of_estimators:
            level_values.append(float(estimator.parameter.level))
            angle_values.append(float(estimator.parameter.angle))
            radius_values.append(float(estimator.parameter.radius))
        return level_values, angle_values, radius_values

    ################################################################################################################
    def mean_estimation(self, level_values, angle_values, radius_values):  # y_train is weight of parameters which are equal to
        if len(level_values) == 0:
            return None

        ave_level = np.average(level_values)
        ave_angle = np.average(angle_values)
        ave_radius = np.average(radius_values)
        new_parameter = Parameter(ave_level, ave_angle, ave_radius)

        return new_parameter

    ####################################################################################################################
    def mode_estimation(self, level_values, angle_values, radius_values):  # y_train is weight of parameters which are equal to

        if level_values != []:

            ave_level = stats.mode(level_values)[0][0]
            ave_angle = stats.mode(angle_values)[0][0]
            ave_radius = stats.mode(radius_values)[0][0]
            new_parameter = Parameter(ave_level, ave_angle, ave_radius)

            return new_parameter
        else:
            return None

    ####################################################################################################################
    @staticmethod
    def median_estimation(level_values, angle_values, radius_values):  # y_train is weight of parameters which are equal to

        if level_values != []:
            ave_level = np.median(level_values)
            ave_angle = np.median(angle_values)
            ave_radius = np.median(radius_values)
            new_parameter = Parameter(ave_level, ave_angle, ave_radius)

            return new_parameter
        else:
            return None

    ###################################################################################################################
    def generation(self, set_of_estimators, unknown_agent,  current_state):


        # if set_of_estimators.Estimators == [] or len(set_of_estimators.Estimators) == 0:
        #     tmp_sim = unknown_agent.choose_target_state.copy()
        #     ##todo: Do I need to have unknown_agent.parameter as input
        #     self.initialisation(unknown_agent.position, unknown_agent.direction, unknown_agent.parameter, tmp_sim)

        # 1. Generating new estimators

        if len(self.bag_of_estimators.radius_bag) > 0:
            random_creation = self.oeata_config.mutation_rate * (self.oeata_config.estimators_length - len(set_of_estimators.Estimators))
            pool_creation = (1 - self.oeata_config.mutation_rate) * (self.oeata_config.estimators_length - len(set_of_estimators.Estimators))
        else:
            random_creation = self.oeata_config.estimators_length - len(set_of_estimators.Estimators)
            pool_creation = 0

        none_count, loop_count, none_threshold = 0, 0, self.oeata_config.estimators_length
        while len(set_of_estimators.Estimators) < self.oeata_config.estimators_length:
            # a. Sampling an estimator
            loop_count += 1
            if loop_count > 1000:
                break

            tmp_param = Parameter()

            if none_count < pool_creation:
                tmp_param.radius = random.choice(self.bag_of_estimators.radius_bag)
                tmp_param.angle = random.choice(self.bag_of_estimators.angles_bag)
                tmp_param.level = random.choice(self.bag_of_estimators.levels_bag)
            else:
                tmp_param = self.oeata_config.fundamental_values.generate_random_parameter()

            # b. Simulating and Filtering the particle
            # i. creating the new particle
            x, y, direction = unknown_agent.position[0], unknown_agent.position[1], unknown_agent.direction

            tmp_agent = Agent('T',set_of_estimators.type, (x, y), direction,
                              tmp_param.radius, tmp_param.angle, tmp_param.level)

            success_in_history = self.check_history(tmp_param, set_of_estimators.type)
            if success_in_history > 0:

                target = get_target_non_adhoc_agent(tmp_agent, current_state)
                estimator = Estimator(tmp_param, current_state.copy(), target,
                                      success_in_history, 0)

                set_of_estimators.Estimators.append(estimator)
                none_count += 1

            del tmp_agent
            del tmp_param

    ###################################################################################################################
    def initialisation(self, unknown_agent_position, unknown_agent_direction,  unknown_agent_radius,unknown_agent_angle
                       ,unknown_agent_level, choose_target_state):
        # 1. Generating initial estimators
        tmp_cts = choose_target_state.copy()
        none_count, none_threshold = 0, 500

        for type in self.oeata_config.fundamental_values.agent_types:
            set_of_estimators = SetOfEstimators(type)

            x, y, direction = unknown_agent_position[0], unknown_agent_position[1], unknown_agent_direction
            tmp_agent = Agent('T', set_of_estimators.type, (x, y), direction,
                              unknown_agent_radius, unknown_agent_angle
                              , unknown_agent_level)

            target = get_target_non_adhoc_agent(tmp_agent, tmp_cts)

            # 5. Adding to the list of estimators
            if target is not (-1, -1):
                agent_param = Parameter(tmp_agent.level, tmp_agent.angle, tmp_agent.radius)
                estimator = Estimator(agent_param, tmp_cts, target, 1, 0)
                set_of_estimators.Estimators.append(estimator)

            tmp_parameter = Parameter()
            while len(set_of_estimators.Estimators) < self.oeata_config.estimators_length:
                del tmp_agent
                if none_count == none_threshold:
                    break
                else:
                    # 2. Random uniform parameter sampling
                    tmp_parameter = self.oeata_config.fundamental_values.generate_random_parameter()

                    # 3. Creating the temporary agent
                    x,y,direction = unknown_agent_position[0], unknown_agent_position[1], unknown_agent_direction
                    tmp_agent = Agent('T',type, (x,y),  direction, tmp_parameter.radius, tmp_parameter.angle,
                                      tmp_parameter.level)

                    # 4. Calculating route
                    tmp_cts = choose_target_state.copy()
                    target = get_target_non_adhoc_agent(tmp_agent, choose_target_state)

                    # 5. Adding to the data set
                    if target is not (-1, -1):
                        agent_param = Parameter(tmp_agent.level, tmp_agent.angle, tmp_agent.radius)
                        estimator = Estimator(agent_param, tmp_cts, target, 1, 0)
                        set_of_estimators.Estimators.append(estimator)
                    else:
                        none_count += 1
            self.all_estimators.append(set_of_estimators)

    ###################################################################################################################
    def estimation(self, set_of_estimators):
        # 1. Getting the last agent parameter estimation
        last_parameters_value = None

        level_values, angle_values, radius_values = self.extract_parameter_values(set_of_estimators.Estimators)
        # 2. Running the estimation method if the train data
        # sets are not empty
        estimated_parameter = None
        if level_values != [] :
            if self.oeata_config.parameter_calculation_mode == 'MEAN':
                estimated_parameter = self.mean_estimation(level_values, angle_values, radius_values)
            if self.oeata_config.parameter_calculation_mode == 'MODE':
                estimated_parameter = self.mode_estimation(level_values, angle_values, radius_values)
            if self.oeata_config.parameter_calculation_mode == 'MEDIAN':
                estimated_parameter = self.median_estimation(level_values, angle_values, radius_values)
        else:
            estimated_parameter = last_parameters_value
            # # 5. Updating the succeeded steps
        succeeded_sum = sum(estimator.success_rate for estimator in set_of_estimators.Estimators)

        if float(self.load_count) == 0.0:
            type_prob = 0.0
        else:
            type_prob = succeeded_sum

        return estimated_parameter, type_prob

    #####################################################################################################
    def update_estimators(self, set_of_estimators, cts_agent,  current_state, just_finished_tasks):
        # 1. Getting the agent to update

        for estimator in set_of_estimators.Estimators:

            for task in just_finished_tasks:
                if estimator.target_task == task.position:

                    tmp_parameter = estimator.parameter
                    x, y = cts_agent.position[0], cts_agent.position[1]
                    direction = cts_agent.direction

                    tmp_agent = Agent('T', set_of_estimators.type, (x, y), direction, tmp_parameter.radius, tmp_parameter.angle,
                                      tmp_parameter.level)

                    target = get_target_non_adhoc_agent(tmp_agent, current_state.copy())

                    if target is not (-1,-1) or current_state.items_left() == 0:
                        estimator.target = target
                        estimator.choose_target_state = current_state.copy()
                    del tmp_agent
    ####################################################################################################################
    def get_agent_index(self):
        return
    ####################################################################################################################
    def check_history(self, parameter, selected_type):

        success_count = 0
        # print 'begin history ---------------------------------------------'
        for hist in self.history_of_tasks:
            # print hist
            old_state = hist.choose_target_state.copy()
            for a in old_state.components['agents']:
                if a.index == self.agent.index:
                    (x, y) = a.position
                    direction = a.direction
                    break


            tmp_agent = Agent('T', selected_type, (x, y), direction, parameter.radius,
                              parameter.angle, parameter.level)

            target = get_target_non_adhoc_agent(tmp_agent, old_state)

            if target == hist.target:
                # print target, hist['loaded_item']
                success_count += 1
        # print 'end history ---------------------------------------------'
        return success_count

    ###################################################################################################################
    def evaluation(self, set_of_estimators, cts_agent, current_state ):
        # 1. Getting the agent to update
        remaining_tasks = 0
        for task in current_state.components['tasks']:
            if not task.completed:
                remaining_tasks +=1


        last_completed_task = cts_agent.smart_parameters['last_completed_task']

        # 3. Running and updating the estimator filter method


        self.load_count += 1
        estimators_to_remove = []
        for estimator in set_of_estimators.Estimators:

            if remaining_tasks != 0: # Is there any item to load
                x, y = cts_agent.position[0], cts_agent.position[1]
                direction = cts_agent.direction

                tmp_agent = Agent('T', set_of_estimators.type, (x, y), direction, estimator.parameter.radius,
                                  estimator.parameter.angle, estimator.parameter.level)

                # 4. Calculating route
                target_task = get_target_non_adhoc_agent(tmp_agent, current_state)

            else:
                break

        # d. Filtering the estimator
            if estimator.target_task == last_completed_task:
                self.bag_of_estimators.levels_bag.append(estimator.parameter.level)
                self.bag_of_estimators.angles_bag.append(estimator.parameter.angle)
                self.bag_of_estimators.radius_bag.append(estimator.parameter.radius)

                estimator.target_task = target_task
                estimator.choose_target_state = current_state.copy()
                estimator.success_rate = self.check_history(estimator.parameter , set_of_estimators.type) + 1
                estimator.failure_rate = 0
            else:
                if int(estimator.failure_rate) > 0:
                    estimators_to_remove.append(estimator)
                else:
                    estimator.failure_rate += 1
                    estimator.success_rate -= 1
                    estimator.target_task = target_task
                    estimator.choose_target_state = current_state.copy()

        # 4. Removing the marked data
        for marked_estimator in estimators_to_remove:
            if marked_estimator in set_of_estimators.Estimators:
                set_of_estimators.Estimators.remove(marked_estimator)

        return
