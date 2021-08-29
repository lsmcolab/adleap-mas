import gc
import numpy as np
import random as rd

from numpy.core.fromnumeric import choose

class Estimator(object):

    def __init__(self, parameters_minmax , type, d):
        self.minmax = parameters_minmax
        self.parameters = []
        for i in range(len(parameters_minmax)):
            min_, max_ = d*parameters_minmax[i][0], (d+1)*parameters_minmax[i][1]
            self.parameters.append(np.random.randint(min_,max_)/d)

        self.type = type
        self.d = d

        self.success_counter = 0
        self.failure_counter = 0
        self.c_e = 0
        self.f_e = 0

        self.predicted_task = None
        self.choose_target_state = None

    def predict_task(self, env, teammate):
        self.choose_target_state = env
        self.predicted_task = env.get_target(teammate.index, self.type, self.parameters)
        return self.predicted_task

    def copy(self):
        copied_estimator = Estimator(self.minmax,self.type,self.d)
        copied_estimator.parameters = np.array([p for p in self.parameters])
        copied_estimator.success_counter = self.success_counter
        copied_estimator.failure_counter = self.failure_counter
        copied_estimator.c_e = self.c_e
        copied_estimator.f_e = self.f_e
        copied_estimator.predicted_task = None
        return copied_estimator


class OEATA(object):

    def __init__(self, initial_state, template_types, parameters_minmax, N=100, xi=2, mr=0.2, d=100, normalise=np.mean):
        # initialising the oeata parameters
        self.template_types = template_types
        self.nparameters = len(parameters_minmax)
        self.parameters_minmax = parameters_minmax
        self.N = N
        self.xi = xi
        self.mr = mr
        self.d = d
        self.normalise = normalise

        # initialising the oeata teammates estimation set
        self.teammate = {}
        self.target_selected = {}
        self.check_teammates_estimation_set(initial_state)

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            tindex = teammate.index
            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}
                self.teammate[tindex]['history'] = []
                self.target_selected[tindex] = False

                # for each type
                for type in self.template_types:
                    # create the estimation set and the bag of estimators
                    self.teammate[tindex][type] = {}
                    self.teammate[tindex][type]['estimation_set'] = []
                    self.teammate[tindex][type]['bag_of_estimators'] = []

                    # initialize the estimation set with N estimators
                    while len(self.teammate[tindex][type]['estimation_set']) < self.N:
                        self.teammate[tindex][type]['estimation_set'].append(\
                            Estimator(self.parameters_minmax, type, self.d))

    def run(self, env):
        # checking if there are new teammates in the environment
        self.check_teammates_estimation_set(env)

        # check the tasks were completed
        just_finished_tasks = set([teammate.smart_parameters['last_completed_task'] \
            for teammate in env.components['agents'] if teammate.smart_parameters['last_completed_task'] != None])

        # running the oeata procedure
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                # if the agent accomplished a task
                if teammate.smart_parameters['last_completed_task'] != None:
                    # 1. Evaluation
                    self.evaluation(env, teammate)

                    # 2. Generation
                    self.generation(env, teammate)
                    
                    self.target_selected[teammate.index] = False

                # 3. Update
                self.update(env, teammate, just_finished_tasks)
        
        return self

    def evaluation(self, env, teammate):
        choose_target_state = None
        completed_task = teammate.smart_parameters['last_completed_task'].copy()
        for type in self.template_types:
            removed = 0

            # saving the choose target state
            if choose_target_state is None and self.teammate[teammate.index][type]['estimation_set'][-1].choose_target_state is not None:
                choose_target_state = self.teammate[teammate.index][type]['estimation_set'][-1].choose_target_state.copy()
            
            for i in range(self.N):
                # verifying if the estimator correctly estimates the task
                # CORRECT
                if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                 self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index == completed_task.index:
                    self.teammate[teammate.index][type]['bag_of_estimators'].append(self.teammate[teammate.index][type]['estimation_set'][i].copy())
                    self.teammate[teammate.index][type]['estimation_set'][i].success_counter += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].c_e = self.teammate[teammate.index][type]['estimation_set'][i].success_counter
                    self.teammate[teammate.index][type]['estimation_set'][i].f_e = 0
                
                # INCORRECT
                else:
                    #updating the failure counter
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].f_e += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].c_e -= 1

                    # checking if the task must be removed
                    if self.teammate[teammate.index][type]['estimation_set'][i].f_e >= self.xi:
                        removed += 1
                        self.teammate[teammate.index][type]['estimation_set'][i].parameters = None
                        gc.collect()
                    
                    #if self.teammate[teammate.index][type]['estimation_set'][i].parameters is not None:
                    #    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                    #    self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()

        # Updating teammate history
        if choose_target_state is not None:
            self.teammate[teammate.index]['history'].append((choose_target_state,completed_task))

    def generation(self, env, teammate):
        for type in self.template_types:
            # 1. Calculating the number of mutations
            n_removed_estimators = sum([self.teammate[teammate.index][type]['estimation_set'][i].parameters is None for i in range(self.N)])
            if len(self.teammate[teammate.index][type]['bag_of_estimators']) > 0:
                nmutations = int(self.mr*n_removed_estimators)
            else:
                nmutations = 0

            # 2. Generating new estimators
            for i in range(self.N):
                if self.teammate[teammate.index][type]['estimation_set'][i].parameters is None:
                    # from the bag
                    if nmutations < 0 and len(self.teammate[teammate.index][type]['bag_of_estimators']) > 1:
                        new_estimator = Estimator(self.parameters_minmax, type, self.d)
                        for n in range(self.nparameters):
                            sampled_estimator = rd.sample(self.teammate[teammate.index][type]['bag_of_estimators'],1)[0]
                            new_estimator.parameters[n] = sampled_estimator.parameters[n]
                        self.teammate[teammate.index][type]['estimation_set'][i] = new_estimator
                    # from the uniform distribution
                    else:
                        self.teammate[teammate.index][type]['estimation_set'][i] = Estimator(self.parameters_minmax, type, self.d)
                        nmutations -= 1
                    
                    # updating the estimator attributes
                    hist_success = self.check_history(self.teammate[teammate.index][type]['estimation_set'][i], teammate, self.teammate[teammate.index]['history'])
                    self.teammate[teammate.index][type]['estimation_set'][i].success_counter = hist_success
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter = 0
                    self.teammate[teammate.index][type]['estimation_set'][i].c_e = hist_success
                    self.teammate[teammate.index][type]['estimation_set'][i].f_e = 0
                    #self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                    #self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()

    def check_history(self, estimator, teammate, history):
        hist_success = 0
        for (state, target) in history:
            p_target = estimator.predict_task(state, teammate)
            if p_target is not None:
                if p_target.index == target.index or\
                 p_target.position == target.position:
                    hist_success += 1
        return hist_success

    def update(self, env, teammate, just_finished_tasks):
        # updating the estimator-predicted task
        just_finished_indexes = [task.index for task in just_finished_tasks]

        if teammate.target is not None and self.target_selected[teammate.index] is False:
            for type in self.template_types:
                for i in range(self.N):
                    self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                    self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()
            self.target_selected[teammate.index] = True

        if teammate.target is  None and self.target_selected[teammate.index] is True:
            for type in self.template_types:
                for i in range(self.N):
                    if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index in just_finished_indexes:
                        self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = None
                        self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = None
            self.target_selected[teammate.index] = False
        
        #for type in self.template_types:
            #for i in range(self.N):
                # if the selected task was accomplished by other agent
                #if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                # self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index in just_finished_indexes:
                #   self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = \
                #         self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                #    self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()

                # or if the estimator did not select any task
                #elif self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is None:
                #    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task =\
                #        self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                #    self.teammate[teammate.index][type]['estimation_set'][i].choose_target_state = env.copy()
                    
    def get_estimation(self, env):
        type_probabilities, estimated_parameters = [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index not in self.teammate.keys() and teammate.index != adhoc_agent.index:
                type_prob = np.array([-1 for i in range(0,len(self.template_types))]) 
                parameter_est = []
                for type in self.template_types:
                    parameter_est.append(np.array([-1 for i in range(0,self.nparameters)]))
                type_probabilities.append(list(type_prob))
                estimated_parameters.append(parameter_est)
                continue

            if teammate.index != adhoc_agent.index:
                # type result
                type_prob = np.zeros(len(self.template_types))
                for i in range(len(self.template_types)):
                    type = self.template_types[i]
                    type_prob[i] = np.sum([self.teammate[teammate.index][type]['estimation_set'][i].c_e \
                                        if self.teammate[teammate.index][type]['estimation_set'][i].c_e > 0 else 0 for i in range(self.N)])

                sum_type_prob = np.sum(type_prob)
                if sum_type_prob != 0:
                    type_prob /= sum_type_prob
                else:
                    type_prob += 1
                    type_prob /= len(self.template_types)

                type_probabilities.append(list(type_prob))

                # parameter result
                parameter_est = []
                for i in range(len(self.template_types)):
                    type = self.template_types[i]
                    parameter_est.append(list(self.normalise([e.parameters for e in self.teammate[teammate.index][type]['estimation_set']],axis=0)))
                estimated_parameters.append(parameter_est)

        return type_probabilities, estimated_parameters

    def sample_state(self, env):
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                selected_type = self.sample_type_for_agent(teammate)
                selected_parameter = self.get_parameter_for_selected_type(teammate,selected_type)

                teammate.type = selected_type
                teammate.set_parameters(selected_parameter)

        return env

    def sample_type_for_agent(self, teammate):
        type_prob = np.zeros(len(self.template_types))
        for i in range(len(self.template_types)):
            type = self.template_types[i]
            type_prob[i] = np.sum([self.teammate[teammate.index][type]['estimation_set'][i].c_e \
                                if self.teammate[teammate.index][type]['estimation_set'][i].c_e > 0 else 0 for i in range(self.N)])

        sum_type_prob = np.sum(type_prob)
        if sum_type_prob != 0:
            type_prob /= sum_type_prob
        else:
            type_prob += 1
            type_prob /= len(self.template_types)
        
        sampled_type = rd.choices(self.template_types,type_prob,k=1)
        return sampled_type[0]

    def get_parameter_for_selected_type(self, teammate, selected_type):
        #parameter_est = self.normalise([e.parameters for e in self.teammate[teammate.index][selected_type]['estimation_set']],axis=0)
        parameter_est = rd.sample(self.teammate[teammate.index][selected_type]['estimation_set'],1)[0].parameters
        return parameter_est
