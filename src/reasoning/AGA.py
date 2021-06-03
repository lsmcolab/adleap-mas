import numpy as np
import sklearn.linear_model
import random
from sklearn.metrics import log_loss
from src.reasoning.fundamentals import Parameter
from src.reasoning.parameter_estimation import ParameterEstimation
import warnings
warnings.filterwarnings("ignore")
class AGAConfig:
    def __init__(self,fundamental_values,grid_size=4,step_size=0.01,decay_step=0.999):
        self.fundamental_values = fundamental_values
        self.step_size = step_size
        self.decay_step = decay_step
        self.grid_size = int(grid_size)


class AGAprocess:
    def __init__(self,aga_config,env):
        self.config = aga_config
        adhoc_agent = env.get_adhoc_agent()
        self.adhoc_index = adhoc_agent.index
        self.previous_state = env

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



            y = [0.96 if (pred_action[i] == actual_action) else 0.04 for i in range(0, len(pred_action))]

            current_estimate = agents.smart_parameters['estimations'].get_parameters_for_selected_type(type)

            new_estimate = self.aga_update(grid,y,current_estimate,2)


            agents.smart_parameters['estimations'].update_type(type,new_estimate,probability_update)

            self.config.step_size*=self.config.decay_step
        #################################################

        self.previous_state = env
        return


    def aga_update(self,X,y,current_estimate,degree=2,univariate=True):
        step_size = self.config.step_size

        if not univariate:
            reg = sklearn.linear_model.LinearRegression()

            reg.fit(np.array(X), y)
            gradient = reg.coef_

            # f_coefficients = np.polynomial.polynomial.polyfit(x_train, y_train,
            #                                                   deg=self.polynomial_degree, full=False)

            current_estimate.radius += self.config.step_size * gradient[0]
            current_estimate.angle += self.config.step_size * gradient[1]
            current_estimate.level += self.config.step_size * gradient[2]
            current_estimate.radius = np.clip(current_estimate.radius, self.config.fundamental_values.radius_min,
                                              self.config.fundamental_values.radius_max)
            current_estimate.angle = np.clip(current_estimate.angle, self.config.fundamental_values.angle_min,
                                             self.config.fundamental_values.angle_max)
            current_estimate.level = np.clip(current_estimate.level, self.config.fundamental_values.level_min,
                                             self.config.fundamental_values.level_max)

            # Not sure if we need this rounding
            # new_parameters.level, new_parameters.angle, new_parameters.radius = \
            #    round(new_parameters.level, 2), round(new_parameters.angle, 2), round(new_parameters.radius, 2)

            return current_estimate

        else:
            parameter_estimate = []
            min_max = []
            min_max.append([self.config.fundamental_values.radius_min,self.config.fundamental_values.radius_max])
            min_max.append([self.config.fundamental_values.angle_min,self.config.fundamental_values.angle_max])
            min_max.append([self.config.fundamental_values.level_min,self.config.fundamental_values.level_max])
            current_est = [current_estimate.radius,current_estimate.angle,current_estimate.level]

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

                # get gradient
                f_poly_deriv = f_poly.deriv()

                current_estimation = current_est[i]

                delta = f_poly_deriv(current_estimation)

                # update parameter
                new_estimation = current_estimation + step_size * delta

                if (new_estimation < p_min):
                    new_estimation = p_min
                if (new_estimation > p_max):
                    new_estimation = p_max

                parameter_estimate.append(new_estimation)
            return Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])
