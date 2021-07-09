import gc
import numpy as np
import random as rd

class Estimator(object):

    def __init__(self, nparameters, type, d):
        self.parameters = np.random.randint(0,d+1,nparameters)/d
        self.type = type
        self.d = d
        self.success_counter = 0
        self.failure_counter = 0
        self.c_e = 0
        self.f_e = 0
        self.predicted_task = None

    def predict_task(self, env, teammate):
        task = env.get_target(teammate.index, self.type, self.parameters)
        return task

    def copy(self):
        copied_estimator = Estimator(len(self.parameters),self.type,self.d)
        copied_estimator.parameters = np.array([p for p in self.parameters])
        copied_estimator.success_counter = self.success_counter
        copied_estimator.failure_counter = self.failure_counter
        copied_estimator.c_e = self.c_e
        copied_estimator.f_e = self.f_e
        copied_estimator.predicted_task = None
        return copied_estimator

class OEATA(object):

    def __init__(self, initial_state, template_types, nparameters, N=100, xi=2, mr=0.2, d=100, normalise=np.mean):
        # initialising the oeata parameters
        self.template_types = template_types
        self.nparameters = nparameters
        self.N = N
        self.xi = xi
        self.mr = mr
        self.d = d
        self.normalise = normalise

        # initialising the oeata teammates estimation set
        self.teammate = {}
        self.check_teammates_estimation_set(initial_state)

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
                    self.teammate[teammate.index][type]['estimation_set'] = []
                    self.teammate[teammate.index][type]['bag_of_estimators'] = []
                    self.teammate[teammate.index]['history'] = []

                    # initialize the estimation set with N estimators
                    while len(self.teammate[teammate.index][type]['estimation_set']) < self.N:
                        self.teammate[teammate.index][type]['estimation_set'].append(\
                            Estimator(self.nparameters, type, self.d))
                        self.teammate[teammate.index][type]['estimation_set'][-1].predicted_task = \
                            self.teammate[teammate.index][type]['estimation_set'][-1].predict_task(env,teammate)

    def run(self, env):
        # checking if there are new teammates in the environment
        self.check_teammates_estimation_set(env)

        # check the tasks were completed
        just_finished_tasks = set([teammate.smart_parameters['last_completed_task'] \
            for teammate in env.components['agents'] if teammate.smart_parameters['last_completed_task'] != None])
        print([t.index for t in just_finished_tasks])
        print("Number of tasks left : ", len([(t.index,t.completed) for t in env.components['tasks']]))

        # running the oeata procedure
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
            if teammate.index != adhoc_agent.index:
                # if the agent accomplished a task
                if teammate.smart_parameters['last_completed_task'] != None:
                    # 1. Evaluation
                    self.evaluation(env.copy(), teammate)

                    # 2. Generation
                    self.generation(env.copy(), teammate)

                # if a task was accomplished by another agent
                if just_finished_tasks:
                    # 3. Update
                    self.update(env.copy(), teammate, just_finished_tasks)
        
        return self

    def evaluation(self, env, teammate):
        for type in self.template_types:
            for i in range(self.N):
                # verifying if the estimator correctly estimates the task
                # CORRECT
                if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and\
                 self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index == teammate.smart_parameters['last_completed_task'].index:
                    self.teammate[teammate.index][type]['bag_of_estimators'].append(self.teammate[teammate.index][type]['estimation_set'][i].copy())
                    self.teammate[teammate.index][type]['estimation_set'][i].success_counter += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].c_e = self.teammate[teammate.index][type]['estimation_set'][i].success_counter
                    self.teammate[teammate.index][type]['estimation_set'][i].f_e = 0

                # INCORRECT
                elif self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None:
                    #updating the failure counter
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].f_e += 1
                    self.teammate[teammate.index][type]['estimation_set'][i].c_e -= 1

                    # checking if the task must be removed
                    if self.teammate[teammate.index][type]['estimation_set'][i].f_e >= self.xi:
                        self.teammate[teammate.index][type]['estimation_set'][i].parameters = None
                        gc.collect()
                        
                # estimating the next task
                if self.teammate[teammate.index][type]['estimation_set'][i].parameters is not None:
                    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)
        
        # 3. Updating teammate history
        self.teammate[teammate.index]['history'].append((env.copy(),teammate.smart_parameters['last_completed_task']))

    def generation(self, env, teammate):
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
                        self.teammate[teammate.index][type]['estimation_set'][i] = Estimator(self.nparameters, type, self.d)
                        nmutations -= 1
                    # from the bag
                    elif len(self.teammate[teammate.index][type]['bag_of_estimators']) > 1:
                        sampled_estimator = rd.sample(self.teammate[teammate.index][type]['bag_of_estimators'],1)[0]
                        self.teammate[teammate.index][type]['estimation_set'][i] = sampled_estimator.copy()
                    else:
                        self.teammate[teammate.index][type]['estimation_set'][i] = Estimator(self.nparameters, type, self.d)
                        nmutations -= 1
                    
                    # updating the estimator attributes
                    hist_success = self.check_history(self.teammate[teammate.index][type]['estimation_set'][i], teammate, self.teammate[teammate.index]['history'])
                    self.teammate[teammate.index][type]['estimation_set'][i].success_counter = hist_success
                    self.teammate[teammate.index][type]['estimation_set'][i].failure_counter = 0
                    self.teammate[teammate.index][type]['estimation_set'][i].c_e = hist_success
                    self.teammate[teammate.index][type]['estimation_set'][i].f_e = 0
                    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)

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
        just_finished_indexes = [task.index for task in just_finished_tasks]
        for type in self.template_types:
            for i in range(self.N):
                if self.teammate[teammate.index][type]['estimation_set'][i].predicted_task is not None and self.teammate[teammate.index][type]['estimation_set'][i].predicted_task.index in just_finished_indexes:
                    self.teammate[teammate.index][type]['estimation_set'][i].predicted_task = self.teammate[teammate.index][type]['estimation_set'][i].predict_task(env, teammate)

    def get_estimation(self, env):
        type_probabilities, estimated_parameters = [], []
    
        adhoc_agent = env.get_adhoc_agent()
        for teammate in env.components['agents']:
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
        parameter_est = self.normalise([e.parameters for e in self.teammate[teammate.index][selected_type]['estimation_set']],axis=0)
        return parameter_est