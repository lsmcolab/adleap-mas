import random
from numpy.random import choice
from src.reasoning.OEATA import HistoryElement
# import agent
from random import sample
from src.reasoning.fundamentals import Parameter
from copy import copy
import logging

logging.basicConfig(filename='parameter_estimation.log', format='%(asctime)s %(message)s', level=logging.DEBUG)


########################################################################################################################
class EstimationHistory: #Keeping the history of type and parameter estimations for each agent
    def __init__(self, a_type):
        self.type = a_type  # Type for which we are doing estimation
        self.type_probability = 0
        self.type_probabilities = []
        self.estimation_history = []

    ####################################################################################################################
    def add_estimation_history(self, probability, new_estimated_parameter):
        self.estimation_history.append(new_estimated_parameter)
        self.type_probabilities.append(probability)

    ####################################################################################################################

    def get_estimation_history(self):
        estimation_history = "["
        for est_hist in self.estimation_history:
            estimation_history += "[" + str(est_hist.level) + "," + str(est_hist.angle) + "," + str(
                est_hist.radius) + "],"

        estimation_history = estimation_history[0:len(estimation_history) - 1]
        estimation_history += "]"
        return estimation_history

    ####################################################################################################################
    def get_last_type_probability(self):
        if len(self.type_probabilities) > 0:
            return self.type_probabilities[-1]
        else:
            return 1.0/5  ##todo: change it to the len of types

    ####################################################################################################################
    def get_last_estimation(self):
        if len(self.estimation_history) > 0:
            return self.estimation_history[-1]
        else:
            return sample(self.train_data.data_set,1)[0]
    
    ####################################################################################################################
    def get_last_action_probability(self):
        if len(self.action_probabilities) > 0:
            return self.action_probabilities[-1]
        else:
            return 0.2

    ####################################################################################################################
    def update_estimation(self, estimation, action_probability):
        self.estimation_history.append(estimation)
        self.action_probabilities.append(action_probability)
########################################################################################################################


class ParameterEstimation:

    def __init__(self,  estimation_config):

        self.estimation_config = estimation_config
        self.learning_data = None
        self.estimation_histories = []

        for t in self.estimation_config.fundamental_values.agent_types:
            e = EstimationHistory(t)
            self.estimation_histories.append(e)

        self.iteration = 0

    ####################################################################################################################
    def get_parameters_for_selected_type(self, selected_type):

        for te in self.estimation_histories:
            if selected_type == te.type:
                return te.get_last_estimation()

    ####################################################################################################################
    def generate_equal_probabilities(self):
        probabilities = []
        for i in range(0, len(self.estimation_histories) - 1):
            te = self.estimation_histories[i]
            probabilities.append(round(1.0 / len(self.estimation_config.fundamental_values.agent_types), 2))

        probabilities.append(round(1.0 - ((len(self.estimation_histories) - 1) * (round(1.0 / len(self.estimation_config.fundamental_values.agent_types), 2))),2))
        return probabilities

    ####################################################################################################################
    def get_sampled_probability(self):

        type_probes = list()
        for te in self.estimation_histories:
            type_probes.append(te.get_last_type_probability())

        selected_type = choice(self.estimation_config.fundamental_values.agent_types, p=type_probes)  # random sampling the action

        return selected_type

#####################################################################################################################
    def normalize_type_probabilities(self):
        # 1. Defining the values
        # print 'Normalizing:',self.l1_estimation.type_probability , self.l2_estimation.type_probability,self.l3_estimation.type_probability, self.l4_estimation.type_probability
        sum_of_probabilities = 0
        type_belief_values = []
        for te in self.estimation_histories:
            type_belief_values.append(te.type_probability)
            sum_of_probabilities += te.type_probability

        # 3. Normalising
        if sum_of_probabilities != 0:
            belief_factor = 1 / float(sum_of_probabilities)
            for te in self.estimation_histories:
                te.type_probability *= belief_factor
                te.type_probabilities.append(te.type_probability)

        else:
            probabilities = self.generate_equal_probabilities()
            for i in range(len(self.estimation_histories)):
                self.estimation_histories[i].type_probability = probabilities[i]
                self.estimation_histories[i].type_probabilities.append(probabilities[i])

    ####################################################################################################################

    def get_highest_type_probability(self):

        highest_probability = -1
        selected_type = ''

        for te in self.estimation_histories:
            tmp_prob = te.get_last_type_probability()
            if tmp_prob > highest_probability:
                highest_probability = tmp_prob
                selected_type = type

        return selected_type

    ####################################################################################################################
    def copy_last_estimation(self, agent_type):

        for te in self.estimation_histories:
            if te.type == agent_type:
               return copy(te.get_last_estimation())

        return None

    ####################################################################################################################
    # Initialisation random values for parameters of each type and probability of actions in time step 0
    def estimation_initialisation(self):

        probabilities = self.generate_equal_probabilities()

        for i in range(0, len(self.estimation_histories)):
            te = self.estimation_histories[i]
            te.add_estimation_history(probabilities[i],
                                      self.estimation_config.fundamental_values.generate_random_parameter())

    ####################################################################################################################
    def parameter_estimation(self, x_train, y_train, agent_type):
        # 1. Getting the last agent parameter estimation
        last_parameters_value = self.copy_last_estimation(agent_type)

        # 2. Running the estimation method if the train data
        # sets are not empty
        estimated_parameter = None
        if x_train != [] and y_train != []:

            if self.parameter_estimation_mode == 'AGA':
                estimated_parameter = self.calculate_gradient_ascent(x_train, y_train, last_parameters_value)
            elif self.parameter_estimation_mode == 'ABU':
                estimated_parameter = self.bayesian_updating(x_train, y_train, last_parameters_value)

        else:
            estimated_parameter = last_parameters_value

        return estimated_parameter

    ####################################################################################################################
    def get_last_selected_type_probability(self,selected_type):

        for te in self.estimation_histories:
            if selected_type == te.type:
                return te.get_last_type_probability()

        return 0

    ####################################################################################################################
    def update_internal_state(self, parameters_estimation, selected_type, unknown_agent, po):
        # 1. Defining the agent to update in main agent point of view
        u_agent = None
        # todo: changggggg
        if not po:  # po is partial observation
            for v_a in unknown_agent.choose_target_state.learning_agents[0].visible_agents:
                if v_a.index == unknown_agent.index:
                    u_agent = v_a

        else:
            memory_agents = unknown_agent.choose_target_state.learning_agents[0].agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    u_agent = m_a
                    break

        # 2. Creating the agents for simulation
        tmp_sim = copy(unknown_agent.choose_target_state)
        (x,y), direction = u_agent.get_position(),u_agent.direction 
        tmp_agent = agent.Agent(x, y, direction,-1, tmp_sim.env_dim, selected_type)
        #tmp_agent = Agent(x, y, direction, -1, tmp_sim.env_dim, selected_type)
        tmp_agent.set_parameters(parameters_estimation)

        # 3. Finding the target
        tmp_agent.visible_agents_items(tmp_sim.items, tmp_sim.non_learning_agents)
        target = tmp_agent.choose_target(tmp_sim.items, tmp_sim.non_learning_agents)

        self.iteration += 1
        return target

    ####################################################################################################################
    def get_train_data(self, selected_type):
        for te in self.estimation_histories:
            if selected_type == te.type:
                return copy(te.train_data)

        return None

    ####################################################################################################################
    def update_train_data(self, unknown_agent, previous_state, current_state, selected_type,loaded_items_list, l_agent, po):
        # 1. Copying the selected type train data
        # unknown_agent = deepcopy(u_a)

        max_suceed_cts = None
        train_data = self.get_train_data(selected_type)
        max_succeed_cts = None
        # 2. Updating th Particles
        type_probability = self.get_last_selected_type_probability(selected_type)
        self.data = train_data.\
                generate_data_for_update_parameter(previous_state,unknown_agent,selected_type, po)

        # 3. Updating the estimation train data
        for te in self.estimation_histories:
            if selected_type == te.type:
                te.train_data = copy(train_data)

        # 4. Extrating and returning the train set
        x_train, y_train = train_data.extract_train_set()
        return x_train, y_train, type_probability , max_succeed_cts

    ####################################################################################################################
    def POMCP_estimation(self,curren_state):
        iteration_max= 100
        max_depth = 100
        particle_filter_numbers = 100

        pomcpe = POMCP_estimation.POMCP(iteration_max, max_depth,particle_filter_numbers)
        estimated_parameter, estimated_type = pomcpe.start_estimation (None,  curren_state)
        return estimated_parameter.tolist(), estimated_type

    ####################################################################################################################
    def process_state_of_the_art(self, unknown_agent,previous_state, current_state,  types,loaded_items_list, l_agent, po=False):
        # 1. Initialising the parameter variables

        if self.parameter_estimation_mode == 'POMCP':
            estimated_parameter, estimated_type = self.POMCP_estimation(current_state)
            estimated_parameter = Parameter(estimated_parameter[0], estimated_parameter[1], estimated_parameter[2])
            for te in self.estimation_histories:
                te.type_probability = 1
                te.estimation_history.append(estimated_parameter)
        else :
            # print '>>>>>',po
            # 2. Estimating the agent type
            for selected_type in types:
                # a. updating the train data for the current state
                x_train, y_train, pf_type_probability, max_succeed_cts = \
                    self.update_train_data(unknown_agent, previous_state, current_state, selected_type,
                                           loaded_items_list, l_agent, po)

                # b. estimating the type with the new train data
                new_parameters_estimation = \
                self.parameter_estimation(x_train, y_train, selected_type)

                # c. considering the new parameter estimation and finding the agent's action probabilities
                if new_parameters_estimation is not None:
                    # i. generating the particle for the selected type

                    tmp_sim = previous_state.copy()
                    x = unknown_agent.previous_agent_status.position[0]
                    y = unknown_agent.previous_agent_status.position[1]
                    direction = unknown_agent.previous_agent_status.direction
                    tmp_agent = agent.Agent(x, y, direction,-1, tmp_sim.env_dim, selected_type)
                    tmp_param = Parameter(new_parameters_estimation.level, new_parameters_estimation.radius,
                                          new_parameters_estimation.angle)

                    # Changing the agent's destination based on new estimated parameter
                    tmp_agent.memory = self.update_internal_state(new_parameters_estimation, selected_type,
                                                                  unknown_agent,po)

                    # Runs a simulator object
                    tmp_agent = tmp_sim.move_a_agent(tmp_agent)
                    action_prob = tmp_agent.get_action_probability(unknown_agent.next_action)
                    # print selected_type, tmp_agent.memory.get_position(),action_prob
                    # print '*******************************************'
                    if action_prob is None:
                        action_prob = 0.2
                    # print action_prob
                    # ii. testing the generated particle and updating the estimation

                    for te in self.estimation_histories:
                        if te.type == selected_type:
                            te.type_probability = action_prob * te.get_last_type_probability()

                            te.update_estimation(new_parameters_estimation, action_prob)

            # d. If a load action was performed, restart the estimation process

        self.normalize_type_probabilities()

    ####################################################################################################################
    def process_oeata(self, unknown_agent, current_state,  l_agent, po=False):
        # 1. Initialising the parameter variables

        cts_agent = None
        if not po:
            cts_agent = copy(l_agent.visible_agents[unknown_agent.index])
        else:
            memory_agents = l_agent.agent_memory
            for m_a in memory_agents:
                if m_a.index == unknown_agent.index:
                    cts_agent = m_a
                    break
        # print 'Start process OEATA'
        for set_of_estimators in self.learning_data.all_estimators:
            # print len( self.learning_data.all_estimators)
            if unknown_agent.next_action == 4:
                self.learning_data.evaluation(set_of_estimators, cts_agent, current_state)
                self.learning_data.generation(set_of_estimators, cts_agent, current_state)
            else:
                self.learning_data.update_estimators(set_of_estimators, cts_agent, current_state)

            new_estimated_parameter, type_probability = self.learning_data.estimation(set_of_estimators)
            for estimation_history in self.estimation_histories:
                if set_of_estimators.type == estimation_history.type:

                    estimation_history.estimation_history.append(new_estimated_parameter)
                    estimation_history.type_probability = type_probability

        # print 'End of Process'
        # te.type_probability = pf_type_probability
        #
        #     # d. If a load action was performed, restart the estimation process
        #todo: change unknown_agent to cts_agent. What is their differences
        if unknown_agent.next_action == 4 and unknown_agent.is_item_nearby(current_state.items) != -1:
            if unknown_agent.choose_target_state != None:
                hist = HistoryElement(unknown_agent.choose_target_state.copy(), copy(unknown_agent.last_loaded_item_pos),
                                      copy(unknown_agent.choose_target_pos),unknown_agent.choose_target_direction)

                self.learning_data.history_of_tasks.append(hist)

            unknown_agent.choose_target_state = current_state.copy()
            unknown_agent.choose_target_pos = unknown_agent.get_position()
            unknown_agent.choose_target_direction = unknown_agent.direction

        self.normalize_type_probabilities()

