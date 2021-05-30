import numpy as np
import sklearn.linear_model
import random
from sklearn.metrics import log_loss
from src.reasoning.fundamentals import Parameter
from src.reasoning.parameter_estimation import ParameterEstimation
import warnings
import sys
warnings.filterwarnings("ignore")


def findMin(polynomial):
    derivative = polynomial.deriv()

    roots = derivative.roots()

    minValue = sys.maxsize

    for r in roots:
        if polynomial(r) < minValue:
            minValue = polynomial(r)

    if polynomial(polynomial.domain[0]) < minValue:
        minValue = polynomial(polynomial.domain[0])

    if polynomial(polynomial.domain[1]) < minValue:
        minValue = polynomial(polynomial.domain[1])

    return minValue

def inversePolynomial(polynomialInput, y):
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

def sampleFromBelief(polynomial, sizeList):
    returnMe = [None] * sizeList

    ## To calculate the CDF, I will first get the integral. The lower part is the lowest possible value for the domain
    ## Given a value x, the CDF will be the integral at x, minus the integral at the lowest possible value.
    dist_integ = polynomial.integ()
    lower_part = dist_integ(polynomial.domain[0])
    cdf = dist_integ.copy()
    cdf.coef[0] = cdf.coef[0] - lower_part

    for s in range(sizeList):
        u = np.random.uniform(0, 1)

        returnMe[s] = inversePolynomial(cdf, u)

    return returnMe


class ABUConfig:
    def __init__(self,fundamental_values,grid_size=4):
        self.fundamental_values = fundamental_values

        self.grid_size = int(grid_size)


class ABUprocess:
    def __init__(self,aga_config,env):
        self.config = aga_config
        adhoc_agent = env.get_adhoc_agent()
        self.adhoc_index = adhoc_agent.index
        self.previous_state = env
        self.iteration = 0
        self.belief_poly = [None] * 3
        for a in env.components['agents']:
            if(a.index != adhoc_agent.index):
                a.smart_parameters['estimations'] = ParameterEstimation(self.config)
                a.smart_parameters['estimations'].estimation_initialisation()
                a.smart_parameters['estimations'].normalize_type_probabilities()


    def update(self,env):
        grid = []
        for i in np.linspace(self.config.fundamental_values.radius_min,self.config.fundamental_values.radius_max,self.config.grid_size):
            for j in np.linspace(self.config.fundamental_values.angle_min,self.config.fundamental_values.angle_max,self.config.grid_size):
                for k in np.linspace(self.config.fundamental_values.level_min,self.config.fundamental_values.level_max,self.config.grid_size):
                    grid.append([i,j,k])

        if(env.episode == 0):
            self.previous_state = env
            return
        #################################################

        for agents in env.components['agents']:
            if(agents.index == self.adhoc_index):
                continue
            pred_action = []
            index = agents.index
            type_index = 0
            actual_action = agents.next_action
            pos = None
            probability_update = 1
            old_env = self.previous_state
            # Meaningful updates occur only if an agent is visible in current and previous state.
            for i in range(0, len(old_env.components["agents"])):
                if (old_env.components["agents"][i].index == index):
                    pos = i

            if (pos == None):
                continue
            type = agents.smart_parameters['estimations'].get_highest_type_probability()

            for [radius, angle, level] in grid:
                env_copy = old_env.copy()
                for copy_agents in env_copy.components['agents']:
                    if (copy_agents.index == index):
                        copy_agents.radius = radius
                        copy_agents.angle = angle
                        copy_agents.level = level
                        copy_agents.target = None
                        copy_agents.type = type
                    elif(copy_agents.index == self.adhoc_index):
                        pass

                    else:
                        copy_agents.radius = random.uniform(self.config.fundamental_values.radius_min, self.config.fundamental_values.radius_max)
                        copy_agents.angle = random.uniform(self.config.fundamental_values.angle_min, self.config.fundamental_values.angle_max)
                        copy_agents.level = random.uniform(self.config.fundamental_values.level_min, self.config.fundamental_values.level_max)
                        copy_agents.target = None
                        copy_agents.type = copy_agents.smart_parameters['estimations'].get_sampled_probability()

                # Can use any step
                env_copy.step(random.sample([i for i in range(0,env.action_space.n)],1)[0])
                pred_action.append(env_copy.components['agents'][pos].next_action)


                if (env_copy.components["agents"][pos].next_action == actual_action):
                    probability_update *=  1.04
                else:
                    probability_update *=  0.96



            y = [0.96 if (pred_action[i] == actual_action) else 0.01 for i in range(0, len(pred_action))]

            current_estimate = agents.smart_parameters['estimations'].get_parameters_for_selected_type(type)

            new_estimate = self.abu_update(grid,y,current_estimate,2)


            agents.smart_parameters['estimations'].update_type(type,new_estimate,probability_update)

        #################################################

        self.previous_state = env
        return


    def abu_update(self,X,y,current_estimate,degree=2,sampling="average"):
        parameter_estimate = []
        min_max = []
        min_max.append([self.config.fundamental_values.radius_min, self.config.fundamental_values.radius_max])
        min_max.append([self.config.fundamental_values.angle_min, self.config.fundamental_values.angle_max])
        min_max.append([self.config.fundamental_values.level_min, self.config.fundamental_values.level_max])
        current_est = [current_estimate.radius, current_estimate.angle, current_estimate.level]


        for i in range(3):
            # Get current independent variables

            current_parameter_set = [elem[i] for elem in X]

            # Obtain the parameter in questions upper and lower limits
            p_min = min_max[i][0]
            p_max = min_max[i][1]

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
            spacing = len(X)
            # Generate equally spaced points, unique to the parameter being modelled
            X_1 = np.linspace(p_min, p_max, spacing)
            y = np.array([g_poly(j) for j in X_1])

            # Future polynomials are modelled using X and y, not D as it's simpler this way. I've left D in for now

            # Fit h
            h_hat_coefficients = np.polynomial.polynomial.polyfit(X_1, y, deg=degree, full=False)

            h_poly = np.polynomial.polynomial.Polynomial(coef=h_hat_coefficients, domain=[p_min, p_max],
                                                         window=[p_min, p_max])

            # "Lift" the polynomial. Perhaps this technique is different than the one in Albrecht and Stone 2017.
            min_h = findMin(h_poly)
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
                x_random = sampleFromBelief(new_belief, 10)
                parameter_estimate.append(np.mean(x_random))

            # Increment iterator
        self.iteration += 1
        new_parameter = Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])



        return new_parameter
