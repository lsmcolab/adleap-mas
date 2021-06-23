import numpy as np
import random as rd
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

class AGA(object):

    def __init__(self, env, template_types, parameters_minmax, grid_size=4, reward_factor=0.04, step_size=0.01, decay_step=0.999, degree=2, univariate=True):
        # initialising the AGA parameters
        self.template_types = template_types
        self.nparameters = len(parameters_minmax)
        self.parameters_minmax = parameters_minmax

        self.grid_size = grid_size
        self.reward_factor = reward_factor
        self.step_size = step_size
        self.decay_step = decay_step
        self.degree = degree
        self.univariate = univariate

        self.previous_state = None

        # initialising the estimation grid
        self.init_estimation_grid()

        # initialising the estimation for the agents
        self.teammate = {}
        self.check_teammates_estimation_set(env)

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            tindex = teammate.index
            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}
                
                # for each type
                for type in self.template_types:
                    # create the estimation set and the bag of estimators
                    self.teammate[teammate.index][type] = {}
                    self.teammate[teammate.index][type]['probability_history'] = [1/len(self.template_types)]
                    self.teammate[teammate.index][type]['parameter_estimation_history'] = \
                        [[rd.uniform(self.parameters_minmax[n][0],self.parameters_minmax[n][1]) for n in range(self.nparameters)]]

    # initialise the estimation grid
    def init_estimation_grid(self):
        self.grid = []
        linear_spaces = [np.linspace(self.parameters_minmax[n][0],self.parameters_minmax[n][0],self.grid_size) for n in range(self.nparameters)]
        for j in range(self.grid_size):
            self.grid.append([linear_spaces[i][j] for i in range(self.nparameters)])
    
    # update the grid given the current environment
    def update(self,env):
        # AGA estimation requires, at least, one previous state in the history to start the estimation
        if self.previous_state is None:
            self.previous_state = env.copy()
            return self
        
        #####
        # START OF AGA ESTIMATION
        #####
        # Performing the AGA estimation
        # - here we will predict the agents actions given a current estimation, so
        # for each agent in the simulation
        adhoc_agent = env.get_adhoc_agent()
        for agent in env.components['agents']:
            pred_actions = []
            probability_update = 1.0
            # - if the agent is not the adhoc agent and it was seen in the previous state
            if agent.index != adhoc_agent.index and self.is_in_the_previous_state(agent):
                # a. pick its best type
                type = self.get_type_with_highest_probability(agent.index)

                # b. for each parameter in the estimation grid, predict the agent action
                for parameters in self.grid:
                    # - setting up an environment copy to evaluate the predictions
                    env_copy = self.previous_state.copy()
                    for copy_agent in env_copy.components['agents']:
                        if copy_agent.index == agent.index:
                            copy_agent.set_parameters(parameters)
                            copy_agent.target = None
                            copy_agent.type = type

                            ag_vector_index = env_copy.components['agents'].index(copy_agent)

                        elif copy_agent.index != adhoc_agent.index:
                            random_parameters = [rd.uniform(self.parameters_minmax[n][0], self.parameters_minmax[n][1]) for n in range(self.nparameters)]
                            copy_agent.set_parameters(random_parameters)
                            copy_agent.target = None
                            copy_agent.type = self.weighted_sample_type(copy_agent.index)

                    # - stepping the simulation to get the teammate action
                    env_copy.step(env.action_space.sample())
                    pred_actions.append(env_copy.components['agents'][ag_vector_index].next_action)

                    # - increase the update factor given correct predictions
                    if (env_copy.components["agents"][ag_vector_index].next_action == agent.next_action):
                        probability_update *=  (1 + self.reward_factor)
                    else:
                        probability_update *=  (1 - self.reward_factor)

                # d. defining the probability update vector
                y = [0.96 if (pred_actions[i] == agent.next_action) else 0.04 for i in range(0, len(pred_actions))]
                current_estimation = self.get_parameters_for_selected_type(agent.index,type)
                new_estimation = self.regression_update(y, current_estimation)

                # e. updating type probabilities 
                current_type_prob = self.teammate[agent.index][type]['probability_history'][-1]
                new_type_probability = (current_type_prob*probability_update)

                last_types_probabilities = np.array([self.teammate[agent.index][t]['probability_history'][-1] for t in self.template_types])
                last_types_probabilities[self.template_types.index(type)] = new_type_probability
                last_types_probabilities /= sum(last_types_probabilities)
                
                for i in range(len(last_types_probabilities)): 
                    self.teammate[agent.index][self.template_types[i]]['probability_history'].append(last_types_probabilities[i])

                # f. adding the new parameter estimation to the history
                self.teammate[agent.index][type]['parameter_estimation_history'].append(new_estimation)

                # g. updating step size
                self.step_size *= self.decay_step
        #####
        # END OF AGA ESTIMATION
        #####
        # Updating the previous state variable
        self.previous_state = env

        return self

    # checks if an agent is in the previous state
    def is_in_the_previous_state(self, agent):
        for i in range(0, len(self.previous_state.components["agents"])):
            if (self.previous_state.components["agents"][i].index == agent.index):
                return True
        return False

    def get_type_with_highest_probability(self,teammate_index):
        last_types_probabilites = [self.teammate[teammate_index][type]['probability_history'][-1] for type in self.template_types]
        return self.template_types[last_types_probabilites.index(max(last_types_probabilites))]

    def weighted_sample_type(self,teammate_index):
        last_types_probabilites = [self.teammate[teammate_index][type]['probability_history'][-1] for type in self.template_types]
        return rd.choices(self.template_types, last_types_probabilites)[0]

    def get_parameters_for_selected_type(self,teammate_index, type):
        parameters = self.teammate[teammate_index][type]['parameter_estimation_history'][-1]
        return parameters

    # AGA regression update
    def regression_update(self,y,current_estimation):
        # if it is an univariate linear regrassion
        if not self.univariate:
            # 1. Initialise the Linear Regression function
            reg = LinearRegression()

            # 2. Fit it to your data
            reg.fit(np.array(self.grid), y)

            # 3. Get the regression coeficient
            gradient = reg.coef_

            # 4. Updating the parameters
            current_estimation += (self.config.step_size * gradient)
            for n in range(self.nparameters):
                current_estimation[n] = np.clip(current_estimation[n], self.parameters_minmax[n][0],self.parameters_minmax[n][1])

            return current_estimation

        # else
        else:
            parameter_estimate = []
            for i in range(self.nparameters):
                # Get current independent variables
                current_parameter_set = [elem[i] for elem in self.grid]

                # Obtain the parameter in questions upper and lower limits
                p_min = self.parameters_minmax[i][0]
                p_max = self.parameters_minmax[i][1]

                # Fit polynomial to the parameter being modelled
                f_poly = np.polynomial.polynomial.polyfit(current_parameter_set, y,
                                                          deg=self.degree, full=False)

                f_poly = np.polynomial.polynomial.Polynomial(coef=f_poly, domain=[p_min, p_max], window=[p_min, p_max])

                # get gradient
                f_poly_deriv = f_poly.deriv()

                # get delta
                delta = f_poly_deriv(current_estimation[i])

                # update parameter
                new_estimation = current_estimation + (self.step_size * delta)
                new_estimation[i] = np.clip(new_estimation[i], self.parameters_minmax[i][0],self.parameters_minmax[i][1])

                print(new_estimation)
                parameter_estimate.append(new_estimation)
            return parameter_estimate

    def get_estimation(self,env):
        types, type_probabilities, parameter_estimation = {}, {}, {}
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                # type result
                type_prob = []
                for type in self.template_types:
                    type_prob.append(self.teammate[teammate.index][type]['probability_history'][-1])

                # parameter result (best type)
                best_type_index = list(type_prob).index(np.max(type_prob))
                best_type = self.template_types[best_type_index]
                parameter_est = self.teammate[teammate.index][best_type]['parameter_estimation_history'][-1]

                # setting the result
                types[teammate.index] = type
                type_probabilities[teammate.index] = list(type_prob)
                parameter_estimation[teammate.index] = parameter_est

        return types, type_probabilities, parameter_estimation