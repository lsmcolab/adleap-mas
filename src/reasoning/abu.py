import itertools
import math
import numpy as np
import random as rd

import warnings
warnings.filterwarnings("ignore")

class ABU(object):

    def __init__(self, env, template_types, parameters_minmax, grid_size=100, reward_factor=0.04, degree=2):
        # initialising the AGA parameters
        self.template_types = template_types
        self.nparameters = len(parameters_minmax)
        self.parameters_minmax = parameters_minmax

        self.grid_size = grid_size
        self.reward_factor = reward_factor
        self.degree = degree

        self.iteration = 0
        self.previous_state = None
        self.belief_poly = [None] * self.nparameters

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
    def init_estimation_grid(self, mode='random'):
        # linear init
        if mode == 'linear':
            mult = int((self.grid_size)**(1/self.nparameters))
            linear_spaces = [np.linspace(self.parameters_minmax[n][0],self.parameters_minmax[n][1],mult) for n in range(self.nparameters)]
            self.grid = list(itertools.product(*linear_spaces))

        # random uniform init
        elif mode == 'random':
            self.grid = np.random.uniform(0, 1, (self.grid_size, self.nparameters))
        
        else:
            raise NotImplemented

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
            # - if the agent is not the adhoc agent and it was seen in the previous state
            if agent.index != adhoc_agent.index and self.is_in_the_previous_state(agent):
                # a. pick its best type
                best_type = self.get_type_with_highest_probability(agent.index)

                # b. predict the action given a test parameter/best type
                pred_actions, probability_update = self.simulate(env, agent, best_type)

                # c. performing the regression update
                y = [0.96 if (pred_actions[i] == agent.next_action) else 0.04 for i in range(0, len(pred_actions))]
                new_estimation = self.abu_update(y)

                # d. updating type probabilities 
                current_type_prob = self.teammate[agent.index][best_type]['probability_history'][-1]
                new_type_probability = (current_type_prob*probability_update)

                last_types_probabilities = np.array([self.teammate[agent.index][t]['probability_history'][-1] for t in self.template_types])
                last_types_probabilities[self.template_types.index(best_type)] = new_type_probability
                last_types_probabilities /= sum(last_types_probabilities)
                
                for i in range(len(last_types_probabilities)): 
                    self.teammate[agent.index][self.template_types[i]]['probability_history'].append(last_types_probabilities[i])

                # e. adding the new parameter estimation to the history
                self.teammate[agent.index][best_type]['parameter_estimation_history'].append(new_estimation)
        #####
        # END OF AGA ESTIMATION
        #####
        # Updating the previous state variable
        self.previous_state = env

        return self

    # predict the agent actions given the current estimation grid
    def simulate(self, env, agent, best_type):
        # For each parameter in the estimation grid, predict the agent action
        pred_actions, probability_update = [], 1.0
        for parameters in self.grid:
            # - setting up an environment copy to evaluate the predictions
            env_copy = self.previous_state.copy()
            adhoc_agent = env_copy.get_adhoc_agent()
            ag_counter = 0
            for copy_agent in env_copy.components['agents']:
                if copy_agent.index == agent.index:
                    copy_agent.set_parameters(parameters)
                    copy_agent.target = None
                    copy_agent.type = best_type

                    ag_vector_index = ag_counter

                elif copy_agent.index != adhoc_agent.index:
                    random_parameters = [rd.uniform(self.parameters_minmax[n][0], self.parameters_minmax[n][1]) for n in range(self.nparameters)]
                    copy_agent.set_parameters(random_parameters)
                    copy_agent.target = None
                    copy_agent.type = self.weighted_sample_type(copy_agent.index)
                
                ag_counter += 1

            # - stepping the simulation to get the teammate action
            env_copy.step(env.action_space.sample())
            pred_actions.append(env_copy.components['agents'][ag_vector_index].next_action)

            # - increase the update factor given correct predictions
            if (env_copy.components["agents"][ag_vector_index].next_action == agent.next_action):
                probability_update *=  (1 + self.reward_factor)
            else:
                probability_update *=  (1 - self.reward_factor)

        return pred_actions, probability_update

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

    def abu_update(self,y,degree=2,sampling="average"):
        parameter_estimate = []
        for i in range(self.nparameters):
            # Get current independent variables
            current_parameter_set = [elem[i] for elem in self.grid]

            # Obtain the parameter in questions upper and lower limits
            p_min = self.parameters_minmax[i][0]
            p_max = self.parameters_minmax[i][1]

            # Fit polynomial to the parameter being modelled
            f_poly = np.polynomial.polynomial.polyfit(current_parameter_set, y,
                                                      deg=degree, full=False)

            f_poly = np.polynomial.polynomial.Polynomial(coef=f_poly, domain=[p_min, p_max], window=[p_min, p_max])

            # Generate prior
            if self.iteration == 0:
                # beliefs = st.uniform.rvs(0, 6AGA_O_2, size=polynomial_degree + 6AGA_O_2)
                beliefs = [0] * (degree + 1)
                beliefs[0] = 1.0 / (p_max - p_min)

                current_belief_poly = np.polynomial.polynomial.Polynomial(coef=beliefs, domain=[p_min, p_max],
                                                                          window=[p_min, p_max])
            else:
                current_belief_poly = self.belief_poly[i]

            # Compute convolution
            g_poly = current_belief_poly * f_poly

            # Collect samples
            # Number of evenly spaced points to compute polynomial at
            # TODO: Not sure why it was polynomial_degree + 6AGA_O_2
            # spacing = polynomial_degree + 6AGA_O_2
            spacing = len(self.grid)
            # Generate equally spaced points, unique to the parameter being modelled
            X_1 = np.linspace(p_min, p_max, spacing)
            y = np.array([g_poly(j) for j in X_1])

            # Future polynomials are modelled using X and y, not D as it's simpler this way. I've left D in for now

            # Fit h
            h_hat_coefficients = np.polynomial.polynomial.polyfit(X_1, y, deg=degree, full=False)

            h_poly = np.polynomial.polynomial.Polynomial(coef=h_hat_coefficients, domain=[p_min, p_max],
                                                         window=[p_min, p_max])

            # "Lift" the polynomial. Perhaps this technique is different than the one in Albrecht and Stone 2017.
            min_h = self.findMin(h_poly)
            if min_h < 0:
                h_poly.coef[0] = h_poly.coef[0] - min_h

            # Integrate h
            integration = h_poly.integ()

            # Compute I
            definite_integral = integration(p_max) - integration(p_min)

            # Update beliefs
            new_belief_coef = np.divide(h_poly.coef, definite_integral)  # returns an array
            new_belief = np.polynomial.polynomial.Polynomial(coef=new_belief_coef, domain=[p_min, p_max],
                                                             window=[p_min, p_max])

            self.belief_poly[i] = new_belief

            # TODO: Not better to derivate and get the roots?
            if sampling == 'MAP':
                # Sample from beliefs
                polynomial_max = 0
                granularity = 1000
                x_vals = np.linspace(p_min, p_max, granularity)
                for j in range(len(x_vals)):
                    proposal = new_belief(x_vals[j])
                    # print('Proposal: {}'.format(proposal))
                    if proposal > polynomial_max:
                        polynomial_max = proposal

                parameter_estimate.append(polynomial_max)

            elif sampling == 'average':
                x_random = self.sampleFromBelief(new_belief, 10)
                parameter_estimate.append(np.mean(x_random))

            # Increment iterator
        self.iteration += 1
        return parameter_estimate


    def findMin(self,polynomial):
        derivative = polynomial.deriv()

        roots = derivative.roots()

        import sys
        minValue = sys.maxsize

        for r in roots:
            if polynomial(r) < minValue:
                minValue = polynomial(r)

        if polynomial(polynomial.domain[0]) < minValue:
            minValue = polynomial(polynomial.domain[0])

        if polynomial(polynomial.domain[1]) < minValue:
            minValue = polynomial(polynomial.domain[1])

        return minValue

    def inversePolynomial(self,polynomialInput, y):
        solutions = list()

        polynomial = polynomialInput.copy()

        polynomial.coef[0] = polynomial.coef[0] - y

        roots = polynomial.roots()

        for r in roots:

            if (r >= polynomial.domain[0] and r <= polynomial.domain[1]):
                if (not (isinstance(r, complex))):
                    solutions.append(r)
                elif (r.imag == 0):
                    solutions.append(r.real)

        ## We should always have one solution for the inverse?
        if (len(solutions) > 1):
            print("Warning! Multiple solutions when sampling for ABU")
        return solutions

    def sampleFromBelief(self,polynomial, sizeList):
        returnMe = [None] * sizeList

        ## To calculate the CDF, I will first get the integral. The lower part is the lowest possible value for the domain
        ## Given a value x, the CDF will be the integral at x, minus the integral at the lowest possible value.
        dist_integ = polynomial.integ()
        lower_part = dist_integ(polynomial.domain[0])
        cdf = dist_integ.copy()
        cdf.coef[0] = cdf.coef[0] - lower_part

        for s in range(sizeList):
            u = np.random.uniform(0, 1)

            returnMe[s] = self.inversePolynomial(cdf, u)

        return returnMe


    def get_estimation(self,env):
        type_probabilities, estimated_parameters = [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                # type result
                type_prob = []
                for type in self.template_types:
                    type_prob.append(self.teammate[teammate.index][type]['probability_history'][-1])
                type_probabilities.append(list(type_prob))

                # parameter result
                parameter_est = []
                for type in self.template_types:
                    parameter_est.append(self.teammate[teammate.index][type]['parameter_estimation_history'][-1])
                estimated_parameters.append(list(parameter_est))

        return type_probabilities, estimated_parameters

    def sample_type_for_agent(self, teammate):
        type_prob = np.zeros(len(self.template_types))
        for i in range(len(self.template_types)):
            type = self.template_types[i]
            type_prob[i] = self.teammate[teammate.index][type]['probability_history'][-1]
        
        sampled_type = rd.choices(self.template_types,type_prob,k=1)
        return sampled_type[0]

    def get_parameter_for_selected_type(self, teammate, selected_type):
        parameter_est = self.teammate[teammate.index][selected_type]['parameter_estimation_history'][-1]
        return parameter_est