import gc
import numpy as np
import random as rd

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

        self.prediction_delay = 0
        self.predicted_task = None

    def predict_task(self, env, teammate):
        self.predicted_task = env.get_target(teammate.index, self.type, self.parameters)

    def parameters_are_equal(self,cmp_parameters):
        for i in range(len(self.parameters)):
            if self.parameters[i] != cmp_parameters[i]:
                return False
        return True

    def copy(self):
        copied_estimator = Estimator(self.minmax,self.type,self.d)
        copied_estimator.parameters = np.array([p for p in self.parameters])
        copied_estimator.success_counter = self.success_counter
        copied_estimator.failure_counter = self.failure_counter
        
        copied_estimator.prediction_delay = self.prediction_delay
        copied_estimator.predicted_task = self.predicted_task.copy() \
                                            if self.predicted_task is not None else None
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
        self.check_teammates(initial_state)

    def check_teammates(self,env):
        # Initialising the bag for the agents, if it is missing
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            # for each teammate
            if teammate.index != adhoc_agent.index\
            and teammate.index not in self.teammate:
                self.add_teammate(teammate,env)
    
    def add_teammate(self, teammate, env):
        tindex = teammate.index
        self.teammate[tindex] = {}

        # for each type
        for type in self.template_types:
            # create the estimation set and the bag of estimators
            self.teammate[tindex][type] = {}
            self.teammate[tindex][type]['estimation_set'] = []
            self.teammate[tindex][type]['bag_of_estimators'] = []

            # initialize the estimation set with N estimators
            while len(self.teammate[tindex][type]['estimation_set']) < self.N:
                # - creating the estimator
                self.teammate[tindex][type]['estimation_set'].append(\
                                            Estimator(self.parameters_minmax, type, self.d))
                                            
                # - adding the estimator to the bag
                self.teammate[tindex][type]['bag_of_estimators'].append(\
                    self.teammate[tindex][type]['estimation_set'][-1].copy())

                # - estimating a task
                self.teammate[tindex][type]['estimation_set'][-1].predict_task(env,teammate)

    def run(self, env):
        # checking if there are new teammates in the environment
        self.check_teammates(env)

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
                    self.generation(teammate)

                # 3. Update
                self.update(env, teammate, just_finished_tasks)
        
        return self

    def evaluation(self, env, teammate):
        completed_task = teammate.smart_parameters['last_completed_task'].copy()
        for type in self.template_types:
            for i in range(self.N):
                # verifying if the estimator correctly estimates the task
                # CORRECT
                predicted_task = self.teammate[teammate.index][type]['estimation_set'][i].predicted_task
                if predicted_task is not None and predicted_task.index == completed_task.index:
                    self.teammate[teammate.index][type]['estimation_set'][i].success_counter += 1
        
                    # adding the evaluated estimator to the bag
                    delay_modf = 0.9**self.teammate[teammate.index][type]['estimation_set'][i].prediction_delay 
                    add_ntimes = int(delay_modf*self.teammate[teammate.index][type]['estimation_set'][i].success_counter) + 1 

                    for j in range(add_ntimes):
                        self.teammate[teammate.index][type]['bag_of_estimators'].append(\
                            self.teammate[teammate.index][type]['estimation_set'][i].copy())
                
                # INCORRECT
                else:
                    #updating the failure counter
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter += 1

                    # marking the estimator to remove
                    c_e = self.teammate[teammate.index][type]['estimation_set'][i].success_counter
                    f_e = self.teammate[teammate.index][type]['estimation_set'][i].failure_counter
                    
                    if (c_e / (f_e + c_e)) <= 0.5:
                        for j in range(len(self.teammate[teammate.index][type]['bag_of_estimators'])-1,0,-1):
                            if self.teammate[teammate.index][type]['estimation_set'][i].parameters_are_equal(\
                             self.teammate[teammate.index][type]['bag_of_estimators'][j].parameters):
                                del self.teammate[teammate.index][type]['bag_of_estimators'][j]

                        self.teammate[teammate.index][type]['estimation_set'][i].parameters = None
                        gc.collect()

                # estimating the next task
                if self.teammate[teammate.index][type]['estimation_set'][i].parameters is not None:
                    self.teammate[teammate.index][type]['estimation_set'][i].prediction_delay = 0
                    self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)

    def generation(self, teammate):
        for type in self.template_types:
            # 1. Calculating the number of mutations
            if len(self.teammate[teammate.index][type]['bag_of_estimators']) > 0:
                n_removed_estimators = sum([self.teammate[teammate.index][type]['estimation_set'][i].parameters is None for i in range(self.N)])
                nmutations = int(self.mr*n_removed_estimators)
            else:
                nmutations = 0

            # 2. Generating new estimators
            for i in range(self.N):
                if self.teammate[teammate.index][type]['estimation_set'][i].parameters is None:
                    # from the uniform distribution
                    if nmutations > 0:
                        self.teammate[teammate.index][type]['estimation_set'][i] = Estimator(self.parameters_minmax, type, self.d)
                        nmutations -= 1

                    # from the bag if there is elements
                    elif len(self.teammate[teammate.index][type]['bag_of_estimators']) > 1:
                        new_estimator = Estimator(self.parameters_minmax, type, self.d)
                        for n in range(self.nparameters):
                            sampled_estimator = rd.sample(self.teammate[teammate.index][type]['bag_of_estimators'],1)[0]
                            new_estimator.parameters[n] = sampled_estimator.parameters[n]
                        self.teammate[teammate.index][type]['estimation_set'][i] = new_estimator.copy()
                        self.teammate[teammate.index][type]['estimation_set'][i].prediction_delay = 0
                        self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = None

                    # else uniform    
                    else:
                        self.teammate[teammate.index][type]['estimation_set'][i] = Estimator(self.parameters_minmax, type, self.d)
                        nmutations -= 1

    def update(self, env, teammate, just_finished_tasks):
        # updating the estimator-predicted task
        just_finished_indexes = [task.index for task in just_finished_tasks]
        for type in self.template_types:
            for i in range(self.N):
                # if the selected task was accomplished by other agent
                if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                 self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index in just_finished_indexes:
                    self.teammate[teammate.index][type]['estimation_set'][i].prediction_delay = 0
                    self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)

                # or if the estimator did not select any task
                elif self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is None:
                    self.teammate[teammate.index][type]['estimation_set'][i].prediction_delay += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
                    
    def get_estimation(self, env):
        type_probabilities, estimated_parameters = [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                # type result
                type_prob = np.zeros(len(self.template_types))
                for i in range(len(self.template_types)):
                    selected_type = self.template_types[i]
                    type_prob[i] = len(self.teammate[teammate.index][selected_type]['bag_of_estimators'])

                norm = np.sum(type_prob)
                for i in range(len(self.template_types)):
                    type_prob[i] /= norm

                type_probabilities.append(list(type_prob))

                # parameter result
                parameter_est = []
                for i in range(len(self.template_types)):
                    type = self.template_types[i]
                    parameter_est.append(list(self.normalise([e.parameters for e in self.teammate[teammate.index][type]['bag_of_estimators']],axis=0)))
                estimated_parameters.append(parameter_est)

        return type_probabilities, estimated_parameters

    def sample_type_for_agent(self, teammate):
        # defining the type probabilities
        type_prob = np.zeros(len(self.template_types))
        for i in range(len(self.template_types)):
            selected_type = self.template_types[i]
            type_prob[i] = len(self.teammate[teammate.index][selected_type]['bag_of_estimators'])

        norm = np.sum(type_prob)
        for i in range(len(self.template_types)):
            type_prob[i] /= norm

        # sampling a type
        sampled_type = rd.choices(self.template_types,type_prob,k=1)
        return sampled_type[0]

    def get_parameter_for_selected_type(self, teammate, selected_type):
        parameter_est = self.normalise([e.parameters for e in self.teammate[teammate.index][selected_type]['bag_of_estimators']],axis=0)
        return parameter_est