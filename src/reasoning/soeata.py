import gc
import numpy as np
import random as rd

class SOEATA(object):

    def __init__(self, initial_state, template_types, parameters_minmax, N=100, xi=2, mr=0.2, d=100, normalise=np.mean):
        # initialising the oeata parameters
        self.ntypes = len(template_types)
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
        self.check_teammates_estimation_set(initial_state)

    def check_teammates_estimation_set(self,env):
        # Initialising the bag for the agents, if it is missing
        for teammate in env.components['agents']:
            tindex = teammate.index
            adhoc_agent = env.get_adhoc_agent()

            if tindex != adhoc_agent.index and tindex not in self.teammate:
                self.teammate[tindex] = {}

                # creating the bag of estimators and types
                self.teammate[tindex] = {}
                self.teammate[tindex]['type_bag'] = [[] for i in range(self.ntypes)]
                self.teammate[tindex]['parameter_bag'] = [[] for i in range(self.nparameters)]
                self.teammate[tindex]['success_counter'] = [{} for i in range(self.nparameters)]
                self.teammate[tindex]['failure_counter'] = [{} for i in range(self.nparameters)]

                for i in range(self.nparameters):     
                    min_, max_ = self.d*self.parameters_minmax[i][0], (self.d+1)*self.parameters_minmax[i][1]
                    for value in np.linspace(min_,max_,self.d):
                        self.teammate[tindex]['success_counter'][value] = 0 
                        self.teammate[tindex]['failure_counter'][value] = 0

                # initialize the estimation set with N random estimators
                self.teammate[tindex]['estimation_set'] = [[] for i in range(self.nparameters)]
                for n in range(self.N):
                    for i in range(self.nparameters):
                        min_, max_ = self.d*self.parameters_minmax[i][0], (self.d+1)*self.parameters_minmax[i][1]
                        estimator = np.random.randint(min_,max_)/self.d
                        self.teammate[tindex]['estimation_set'].append(estimator)
                
                # - estimation by iteration
                self.teammate[tindex]['estimation_by_iteration'] = []

    def run(self, env):
        # checking if there are new teammates in the environment
        self.check_teammates_estimation_set(env)

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

                # 3. Update
                self.update(env, teammate)
        
        return self

    def evaluation(self, env, teammate):
        completed_task = teammate.smart_parameters['last_completed_task']
        
        
        return

    def generation(self, env, teammate):

        return

    def check_history(self, estimator, teammate, history):

        return

    def update(self, env, teammate):
        # 1. Creating the vector of just finished tasks
        just_finished_tasks = set([teammate.smart_parameters['last_completed_task'] \
            for teammate in env.components['agents'] if teammate.smart_parameters['last_completed_task'] != None])

        # 2.

        return
                    
    def get_estimation(self, env):
        type_probabilities, estimated_parameters = None, None

        return type_probabilities, estimated_parameters

    def sample_state(self, env):

        return env

    def sample_type_for_agent(self, teammate):
        sampled_type = None

        return sampled_type

    def get_parameter_for_selected_type(self, teammate, selected_type):
        parameter_est = None

        return parameter_est