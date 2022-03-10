from importlib import import_module
from copy import deepcopy
import gym
from gym import error,spaces
from gym.envs.classic_control.rendering import LineWidth
import numpy as np
import random as rd
# Necessary Base Classes
from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

'''
Make sure that all __init__.py/setup.py files are updated correspondingly
'''

class Agent(AdhocAgent):
    # Index : Unique Identifier
    # Type : Reasoning/Learning/Planning Type , for ex : POMCP
    def __init__(self,index,type,*args):
        super(Agent,self).__init__(index,type)
        # Base Class will initialise self.index and self.type.
        # Check out the Base Class at src/envs/AdhocReasoningEnv
        # self.smart_parameters is used to save information for the agent

        # Initialise required details
        self.args = args


    # Customised copy. If not implemented, deep copy of Agent is made. 
    # This might give issue with belief trees. 
    def copy(self):
        copy_agent = Agent(self.index,self.position,self.args)
        return copy_agent

    # define other functions, if necessary. But only these 2 are necessary. 
    

# Completely Optional
class ExtraComponent():
    def __init__(self,*args):
        self.args = args
    
    # It is suggested to create copy method for all components of env
    def copy(self):
        return ExtraComponent(self.args)



def end_condition(state):
    # state can be np array, dictionary, list etc. 
    # Check if state is terminal
    return True        
        
def reward(state,next_state):
    # Reward for transition
    return 0

def template_transition(action,real_env):
    agent = real_env.get_adhoc_agent()
    agent.next_action = action

    # If there are multiple agents, we can control action of only the 
    # adhoc agent. The actions of other agents have to be set within this 
    # function. For reference : Look at src/envs/LevelForaginEnv.py

    
    next_state,info = do_action(real_env)    

# This function is just for convenience
def do_action(env):
    # Make relevant changes to the environment and agents
    # Any intermediate reward can be saved in the info dictionary
    # After making changes to env ,
    next_state = update(env)    
    info = {}

    # any key of info with 'reward' as substring is considered as part of reward
    # refer src/envs/AdhocReasoningEnv.py 
    return next_state,env

def update(env):
    # Do any other required updates
    return env.state

def environment_transformation(copied_env):
    # This function converts the copied_env to partially observable
    # state, by masking information that reasoning agent cannot know
    # copied_env is of type : TemplateEnv
    return copied_env    


class TemplateEnvState(spaces.Space):
    def __init__(self,n):
        super(TemplateEnvState,self).__init__(shape=n,dtype=np.float64)
    def sample(self,seed=None):
        return

class TemplateEnv(AdhocReasoningEnv):
    def __init__(self,components,*args,display=False):
        self.viewer = None # For rendering
        self.display = display

        # To represent initial and final condition
        state_set = StateSet(TemplateEnvState,end_condition)
        
        transition_function = template_transition
        reward_function = reward
        observation_space = environment_transformation
        action_space = spaces.Discrete() # Define This. Use appropriate subclass


        super(TemplateEnv,self).__init(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)

        # Set initial state
        self.state_set.initial_components = self.copy_components(components)
        # Set initial state. Type : list,dict, or other simple data types
        self.state_set.initial_state = {}

    # Import helper function for reasoning type
    def import_method(self, agent_type):
        from importlib import import_module
        try:
            # Make separate folder if required
            module = import_module('src.reasoning.template.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = TemplateEnv(components, self.args)
        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env

    # MCTS Helper
    def state_is_equal(self,state):
        # state is same type as self.state

        return True
    
    # POMCP Helper
    def observation_is_equal(self,obs):
        # obs : type TemplateEnv
        return True
    
    def sample_nstate(self, agent, n, sample_a=0.1, sample_t=0.1, n_sample=10):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent,sample_a,sample_t,n_sample))
        return sampled_states

    # Provide List of actions
    def get_actions_list(self):
        # Set this manually, if required
        no_of_actions = self.action_space.n
        return [i for i in range(0,no_of_actions)]

    def get_adhoc_agent(self):
        # This is required if there are multiple agents in components
        adhoc_index = self.components['adhoc_agent_index']
        return self.components['agents'][adhoc_index]
    
    def render(self,mode='human',sleep_=0.5):
        if not self.display:
            return
        
        return

 
