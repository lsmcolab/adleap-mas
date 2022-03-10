from copy import deepcopy
import gym
from gym import spaces
import numpy as np
import random 

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

class Agent(AdhocAgent):
    def __init__(self,index,type="random"):
        super(Agent,self).__init__(index,type)
        self.type = type
        self.index = index
    
    def copy(self):
        copy_agent = Agent(self.index,self.type)
        return copy_agent

def end_condition(state):
    assert state['tiger'] in ['left','right']
    return False

def update(env,obs):
    env.state['obs'] = obs
    return env.state

def listen(tiger_position,missheard_p=0.15):
    coin = np.random.random()
    if coin > missheard_p:
        obs = tiger_position
    else:
        obs = 'left' if tiger_position == 'right' else 'right'
    return obs

def do_action(env):
    action = env.components['agents'][0].next_action
    assert env.pos in ['left','right']
    info = {}

    obs = listen(env.pos) if action == 2 else None
    info['reward'] = reward_intermediate(env.state,action)
    if action<2:
        env.reset_tiger()
    env.state['obs'] = obs


    return env.state,info

def tiger_transition(action,real_env):
    real_env.components['agents'][0].next_action = action
    next_state,info = do_action(real_env)
    return next_state , info

def reward(state,next_state):
    return 0

def reward_intermediate(state,action):
    if action==2:
        return -1
    if state['tiger']=='right' and action==0:
        return 10
    if state['tiger'] == 'left' and action==1:
        return 10

    return -100

def environment_transformation(copied_env):
    copied_env.state['tiger'] = None
    copied_env.pos = None
    return copied_env

class TigerEnvState(spaces.Space):
    def __init__(self):
        super(TigerEnvState,self).__init__(shape=(3,1),dtype=np.float64)

    def sample(self):
        state = {'tiger' : None, 'agent' : None, 'obs' : None}
        return state

class TigerEnv(AdhocReasoningEnv):
    colors = { \
        'red': (1.0, 0.0, 0.0), \
        'darkred': (0.5, 0.0, 0.0), \
        'green': (0.0, 1.0, 0.0), \
        'darkgreen': (0.0, 0.5, 0.0), \
        'blue': (0.0, 0.0, 1.0), \
        'darkblue': (0.0, 0.0, 0.5), \
        'cyan': (0.0, 1.0, 1.0), \
        'darkcyan': (0.0, 0.5, 0.5), \
        'magenta': (1.0, 0.0, 1.0), \
        'darkmagenta': (0.5, 0.0, 0.5), \
        'yellow': (1.0, 1.0, 0.0), \
        'darkyellow': (0.5, 0.5, 0.0), \
        'brown': (0, 0.2, 0.2), \
        'white': (1.0, 1.0, 1.0), \
        'lightgrey': (0.8, 0.8, 0.8), \
        'darkgrey': (0.4, 0.4, 0.4), \
        'black': (0.0, 0.0, 0.0)
    }

    action_dict = {\
        0:'left',
        1:'right',
        2:'listen'
    }

    def __init__(self,components,tiger_pos,display=False):
        assert ('agents' in components.keys())
        self.pos = tiger_pos
        self.viewer = None
        self.display = display 
        state_set = StateSet(TigerEnvState,end_condition=end_condition)
        action_space = spaces.Discrete(3)
        reward_function = reward
        observation_space = environment_transformation
        transition_function = tiger_transition
        super(TigerEnv,self).__init__(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)
        
        self.state_set.initial_components = self.copy_components(components) 
        self.state_set.initial_state = {'tiger':None,'obs':None}
        self.state_set.initial_state['tiger'] = self.pos
    

    def reset_tiger(self):
        if self.pos is not None:
            self.pos = random.sample(['left','right'],1)[0]
            self.state['tiger'] = self.pos

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = TigerEnv(components,tiger_pos= self.pos)
        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode
        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env
 
     # The environment is partially observable by definition
    def state_is_equal(self,state):    
        return (state['obs']==self.state['obs'])

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.smartfirebrigade.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def observation_is_equal(self,obs):
        state = obs.state
        return self.state_is_equal(state)

    def sample_state(self,agent):
        u_env = self.copy()
        obs = u_env.get_observation()
        vals = ['left','right']
        obs.pos = random.sample(vals,1)[0]
        obs.state['tiger'] = obs.pos
        return obs
    
    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def get_actions_list(self):
        return [i for i in range(0,3)]

    def get_adhoc_agent(self):
        return self.components['agents'][0]

    def render(self,mode='human',sleep_=0.5):
        if not self.display:
            return
        return
    
    
       