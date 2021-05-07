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
    # Initialisation random values for parameters of each type and probability of actions in time step 0
    def estimation_initialisation(self):

        probabilities = self.generate_equal_probabilities()

        for i in range(0, len(self.estimation_histories)):
            te = self.estimation_histories[i]
            te.add_estimation_history(probabilities[i],
                                      self.estimation_config.fundamental_values.generate_random_parameter())


    ####################################################################################################################
    def get_last_selected_type_probability(self,selected_type):

        for te in self.estimation_histories:
            if selected_type == te.type:
                return te.get_last_type_probability()

        return 0


