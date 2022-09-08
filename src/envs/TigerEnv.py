from copy import deepcopy
from gym import spaces
import numpy as np
import random 

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

MISSHEARD_P = 0.15

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0):
    _, scenario_id = load_default_scenario_components(scenario_id)
    components = {"agents":[Agent(index= 0, type= method)]}
    env = TigerEnv(components=components,tiger_pos=random.choice(['left','right']),display=False)  
    return env, scenario_id

def load_default_scenario_components(scenario_id):
    if scenario_id >= 1:
        print('There is no different scenarios for the Tiger problem. Setting scenario_id to 0.')
        scenario_id = 0
    return None, scenario_id

"""
    Support classes
"""
class Agent(AdhocAgent):
    def __init__(self,index,type="random"):
        super(Agent,self).__init__(index,type)
        self.type = type
        self.index = index
    
    def copy(self):
        copy_agent = Agent(self.index,self.type)
        return copy_agent

class TigerEnvState(spaces.Space):
    def __init__(self):
        super(TigerEnvState,self).__init__(dtype=str)
    
    def sample(self):
        return {'tiger_pos':random.choice(['left','right']),\
                'action':random.choice(['left','right','listen']),\
                'obs':random.choice(['noise_left','noise_right'])}

"""
    Customising the Tiger Env
"""
def end_condition(env):
    return env.state['action'] in ['left','right']

def listen(env,missheard_p=MISSHEARD_P):
    tiger_position = env.state['tiger_pos']

    # if we don't have any information about the tiger
    if tiger_position is None:
        obs = random.choice(['noise_left','noise_right'])
        return obs
        
    coin = np.random.random()
    if coin > missheard_p:
        obs = 'noise_'+tiger_position
    else:
        obs = 'noise_left' if tiger_position == 'right' else 'noise_right'

    return obs

def reward(state,next_state):
    return 0

def reward_intermediate(env,action):
    if env.simulation:
        tiger_pos = env.state['tiger_pos']
    else:
        tiger_pos = env.tiger_pos

    if env.action_dict[action]=='listen':
        return -1
    if tiger_pos == 'left'  and env.action_dict[action]=='right':
        return 10
    if tiger_pos == 'right' and env.action_dict[action]=='left':
        return 10

    return -100

def do_action(env):
    info = {}
    action = env.components['agents'][0].next_action
    env.state['action'] = env.action_dict[action]

    # if listen
    if env.state['action'] == 'listen':
        obs = listen(env)
        env.state['obs'] = obs
        
    info['reward'] = reward_intermediate(env,action)
    return env,info

def tiger_transition(action,real_env):
    real_env.components['agents'][0].next_action = action
    next_state,info = do_action(real_env)
    return next_state , info

def environment_transformation(copied_env):
    return copied_env

"""
    Tiger Environments 
"""

class TigerEnv(AdhocReasoningEnv):

    action_dict = {\
        0:'left',
        1:'right',
        2:'listen'
    }

    observation_dict = {\
        0:'noise_left',
        1:'noise_right'
    }

    def __init__(self,components,tiger_pos,display=False):
        self.viewer = None
        self.display = display 

        ###
        # Env settings
        ###
        self.tiger_pos = tiger_pos
        self.state = {'tiger_pos':tiger_pos,'action':None,'obs':None}

        state_set = StateSet(TigerEnvState,end_condition=end_condition)
        action_space = spaces.Discrete(3)
        reward_function = reward
        observation_space = environment_transformation
        transition_function = tiger_transition

        ###
        # Initialising the env
        ###
        super(TigerEnv,self).__init__(state_set, transition_function,\
            action_space, reward_function, observation_space, components)
        
        self.state_set.initial_components = self.copy_components(components) 
        self.state_set.initial_state = {'tiger_pos':tiger_pos,'action':None,'obs':None}

    def state_is_equal(self,state):
        return self.state['tiger_pos'] == state.state['tiger_pos']

    def observation_is_equal(self,obs):
        return self.state['obs'] == obs.state['obs']

    def sample_state(self,agent):
        u_env = self.copy()
        tpos = random.choice(['left','right'])
        u_env.state['tiger_pos'] = tpos
        u_env.tiger_pos = tpos
        return u_env
    
    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            s = self.sample_state(agent)
            sampled_states.append(s)
        return sampled_states

    def get_trans_p(self,action):
        if self.action_dict[action] == 'listen':
            return [self,1]
        return [self,0.5]

    def get_obs_p(self,action):
        if action is None:
            return [self,0.5]

        if self.action_dict[action] == 'listen':
            if self.state['obs'] == None:
                env = self.get_observation()
                self.state['obs'] = env.state['obs']
            if self.state['obs'] == 'noise_'+ self.state['tiger_pos']:
                return [self,(1-MISSHEARD_P)]
            else:
                return [self,(MISSHEARD_P)]

        return [self,0.5]

    def get_actions_list(self):
        return [i for i in range(0,len(self.action_dict))]

    def get_observations_list(self):
        return [i for i in range(0,len(self.observation_dict))]

    def get_adhoc_agent(self):
        return self.components['agents'][0]

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = TigerEnv(components,tiger_pos=self.state['tiger_pos'])

        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.smartfirebrigade.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def render(self,mode='human',sleep_=0.5):
        return False
    