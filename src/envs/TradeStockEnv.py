from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random as rd

from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Support classes
"""
class TSAgent(AdhocAgent):

    def __init__(self,index,atype):
        super(TSAgent,self).__init__(index,atype)
        self.type = atype

    def copy(self):
        cp_agent = TSAgent(self.index,self.type)
        return cp_agent

"""
    Customising the TradeStock Env
"""
def end_condition(state):
    return state.episode == len(state.train) + len(state.test)


def do_action(env,action):
    info = {}
    # If there is no action
    if action is None:
        return env,info
    # Else normalise it
    else:
        action = env.normalise_action(action)
    # If position is open
    if env.state['position'] == 1:
        if (env.state['spread'] > env.stop_loss or env.state['spread'] < -env.stop_loss): #stoploss
            env.state['position'] = 0
            env.state['balance'] += (env.state['stockA'] * env.state['numstockA']) + (env.state['stockB']*env.state['numstockB'])
            env.state['numstockA'] = 0
            env.state['numstockB'] = 0
    
        elif env.state['mean'] !=0 :
            env.state['position'] = 0
            env.state['balance'] += (env.state['stockA'] * env.state['numstockA']) + (env.state['stockB']*env.state['numstockB'])
            env.state['numstockA'] = 0
            env.state['numstockB'] = 0
        else:
            pass

    elif env.state['position'] == 0: #Position is closed
        if env.state['spread'] > action: #Going Long stock A and short stock B
            env.state['position'] = 1
            env.state['entry_level'] = env.state['spread']
            env.state['balance'] += -env.state['stockA'] + abs(env.state['beta'])*env.state['stockB']
            env.state['numstockA'] += 1
            env.state['numstockB'] = -abs(env.state['beta'])
            
        elif env.state['spread'] < -action: #Going Long stock B and short stock A
            env.state['position'] = 1
            env.state['entry_level'] = env.state['spread']
            env.state['balance'] += env.state['stockA'] -  abs(env.state['beta'])*env.state['stockB']
            env.state['numstockA'] += -1
            env.state['numstockB'] = abs(env.state['beta'])
                        
    env.state = {
        'spread':env.test[env.episode+1],
        'stockA': env.priceA[env.episode+1 +len(env.train)], #Price series of A
        'stockB': env.priceB[env.episode+1 + len(env.train)], #Price series of B
        'balance': env.state['balance'], # Cash
        'position': env.state['position'],
        'entry_level': env.state['entry_level'],
        'mean':env.state['mean'],
        'numstockA':env.state['numstockA'],
        'numstockB':env.state['numstockB'],
        'beta':env.state['beta']
        }
    return env,info

def tradestock_transition(action, real_env):
    real_env.components['agent'].next_action = action
    next_state,info = do_action(real_env,action)
    return next_state, info


# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return (state['stockA'] * next_state.state['numstockA']) + (state['stockB']*next_state.state['numstockB']) + next_state.state['balance']


# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    return copied_env


"""
    Trade Stock Environments 
"""
class TradeStockEnv(AdhocReasoningEnv):

    stop_loss = 2.0
    # action space
    action_space_max_value = 0.75*stop_loss
    action_space_size = 5
    action_dict = {}
    for i in range(action_space_size):
        action_dict[i] = str(i+1)
    
    def normalise_action(self,action):
        return -float((action+1)* (self.action_space_max_value / self.action_space_size))
    
    # agents color
    agents_color = {
        'mcts': 'red',
        'pomcp': 'yellow',
        'ibpomcp':'blue',
        'rhopomcp':'cyan'
    }

    def __init__(self, components,display=False):
        ###
        # Env Settings
        ###
        state_set = StateSet(spaces.Dict(
            {'spread':spaces.Box(0,np.inf,(1,)),
             'stockA':spaces.Box(0,np.inf,(1,)),
             'stockB':spaces.Box(0,np.inf,(1,)),
             'balance':spaces.Box(0,np.inf,(1,)),
             'position':spaces.Discrete(1),
             'entry_level':spaces.Box(0,np.inf,(1,)),
             'mean': spaces.Discrete(1),
             'numstockA': spaces.Discrete(1),
             'numstockB': spaces.Discrete(1),
             'beta':spaces.Box(0,np.inf,(1,))
             }
        ), end_condition)
        transition_function = tradestock_transition
        action_space = spaces.Discrete(self.action_space_size)
        reward_function = reward
        observation_space = environment_transformation


        ###
        # Initialising the env
        ###
        super(TradeStockEnv, self).__init__(state_set, \
                                               transition_function, action_space, reward_function, \
                                               observation_space, components)

        # Loading initial data
        self.test = components['test']
        self.train = components['train']
        self.priceA = components['priceA']
        self.priceB = components['priceB']
        self.mean = components['mean']
        self.beta = components['beta']

        self.min_price = min(self.train)
        self.max_price = max(self.train)

        # Setting the inital state
        self.state_set.initial_state = {
            'spread':self.test[0],
            'stockA':self.priceA[0],
            'stockB':self.priceB[0],
            'balance':500,
            'position': 0,
            'entry_level':0,
            'mean':self.mean,
            'numstockA': 0,
            'numstockB': 0,
            'beta': self.beta
        }

        # Setting the initial components
        self.state_set.initial_components = self.copy_components(components)

        ###
        # Setting graphical interface
        ###
        self.fig = None
        self.display = display
        self.screen = None
        self.render_mode = "human"
        self.render_sleep = 0.5
        self.clock = None
        self.renderer = None
        if self.display:
            if self.renderer is None:
                try:
                    from gym.error import DependencyNotInstalled
                    from gym.utils.renderer import Renderer
                except ImportError:
                    raise DependencyNotInstalled(
                        "pygame is not installed, run `pip install gym[classic_control]`"
                    )
                self.renderer = Renderer(self.render_mode, self._render)

    def import_method(self, agent_type):
        from importlib import import_module
        module = import_module('src.reasoning.'+agent_type)
        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        copied_env = TradeStockEnv(self.components, self.display)
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.renderer = self.renderer
        copied_env.simulation = self.simulation

        # Graph
        if not copied_env.simulation:
            copied_env.fig = self.fig
            copied_env.ax = self.ax
            copied_env.hl = self.hl
            copied_env.upper_threshold = self.upper_threshold
            copied_env.lower_threshold = self.lower_threshold

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = self.copy_components(self.state)
        return copied_env

    def get_actions_list(self):
         actions_list = []
         for key in self.action_dict.keys():
            actions_list.append(key)
         return actions_list

    def get_adhoc_agent(self):
        return self.components['agent']

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]
        
    def state_is_equal(self, state):
        return True

    def observation_is_equal(self, obs):
        return True

    def sample_state(self, agent):
        # 1. Defining the base simulation
        u_env = self.copy()
        u_env.state['price'] = rd.uniform(u_env.min_price,u_env.max_price)
        return u_env

    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def render(self):
        return self.renderer.get_renders()
        
    def _render(self, mode="human"):
        ###
        # Plotting
        ###          
        if self.display and self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(12,6))
            self.x = range(len(self.test)+len(self.train))
            self.ax = self.fig.add_subplot(111)
            self.hl, = self.ax.plot(self.x[:len(self.train)], self.train,label='price')  
            self.upper_threshold, = self.ax.plot(\
                self.x[:len(self.train)], [None for i in range(len(self.train))], color='r', linestyle='--')
            self.lower_threshold, = self.ax.plot(\
                self.x[:len(self.train)], [None for i in range(len(self.train))], color='r', linestyle='--',label='position threshold')
            plt.axvline(x = len(self.train)+2, color = 'g', linestyle='--',label='trade begin')
        else:
            self.hl.set_xdata(np.append(self.hl.get_xdata(),self.x[(len(self.train))+(self.episode+1)]))
            self.hl.set_ydata(np.append(self.hl.get_ydata(),self.test[self.episode]))

            action = self.components['agent'].next_action
            self.upper_threshold.set_xdata(np.append(self.upper_threshold.get_xdata(),self.x[(len(self.train))+(self.episode+1)]))
            self.upper_threshold.set_ydata(np.append(self.upper_threshold.get_ydata(),\
                self.normalise_action(action) if action is not None else None))

            self.lower_threshold.set_xdata(np.append(self.lower_threshold.get_xdata(),self.x[(len(self.train))+(self.episode+1)]))
            self.lower_threshold.set_ydata(np.append(self.lower_threshold.get_ydata(),\
                -self.normalise_action(action) if action is not None else None))
        self.autoscale_y()
        plt.legend(loc='upper left')
        plt.xlim((len(self.train)-20,len(self.train)+self.episode+100))
        plt.draw()
        self.fig.canvas.flush_events()
        #time.sleep(0.1)

    def autoscale_y(self,margin=0.1):
        def get_bottom_top(line):
            xd,yd = line.get_xdata(), line.get_ydata()
            lo,hi = self.ax.get_xlim()
            y_displayed = yd[((xd>lo) & (xd<hi))]
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed)-margin*h
            top = np.max(y_displayed)+margin*h
            return bot,top

        bot,top = np.inf, -np.inf
        new_bot, new_top = get_bottom_top(self.hl)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top
        
        if bot is not np.nan and bot != np.inf and\
            top is not np.nan and top != np.inf:
                plt.ylim(bot,top)