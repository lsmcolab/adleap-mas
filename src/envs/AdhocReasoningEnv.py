from copy import deepcopy
import gym
from gym import error, spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding
from inspect import isfunction
import numpy as np
from typing import Callable
from warnings import warn

class AdhocAgent:

    def __init__(self,index,atype=None):
        self.index = index
        self.type = atype
        
        self.next_action = None
        self.target = None
        self.smart_parameters = {}
    
    def copy(self):
        return AdhocAgent(self.index,self.type)

class StateSet:

    def __init__(self,state_representation,end_condition):
        # Setting state form in the space
        if issubclass(type(state_representation),spaces.Space):
            self.state_representation = state_representation
        else:
            raise ValueError("argument \"form\" must be a "+str(spaces.Space)+\
                                                " instance or inherit it.")
        
        # Defining the initial state
        self.initial_state = None
        self.initial_components = None

        # Setting the end condition to define the terminal states
        if isfunction(end_condition):
            self.end_condition = end_condition
        else:
            raise ValueError("argument \"end_condition\" must be a function.")

        warn("do not forget to manually set the initial state and initial components.",UserWarning)

    def is_final_state(self,state):
        if self.end_condition(state):
            return True
        else:
            return False


class AdhocReasoningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,state_set,transition_function,action_space,\
                    reward_function,observation_space,components):
        # Initialise the Ad-hoc Reasoning Environment
        super(AdhocReasoningEnv, self).__init__()
        self.viewer = None
        self.state = None
        self.episode = 0

        # Setting the Markovian components for the environment
        if isinstance(state_set,StateSet):
            self.state_set = state_set
        else:
            raise ValueError("argument 1 \"state_set\" must be a "+\
                                            str(StateSet)+" instance.")

        if isfunction(transition_function):
            if transition_function.__code__.co_argcount != 2:
                raise ValueError("argument 2 \"transition_function\" "+\
                 "must be a function with 2 argument: the environment and"+\
                                        "the action of an agent (integer).")
            self.transition_function = transition_function
        else:
            raise ValueError("argument 2 \"transition_function\" "+\
                                                "must be a function.")

        if issubclass(type(action_space),spaces.Space):
            self.action_space = action_space
        else:
            raise ValueError("argument 3 \"action_space\" must be a "+\
                            str(spaces.Space)+" instance or inherit it.")

        if isfunction(reward_function):
            self.reward_function = reward_function
        else:
            raise ValueError("argument 4 \"reward_function\" "+\
                                            "must be a function.")

        if isfunction(observation_space):
            self.observation_space = observation_space
        else:
            raise ValueError("argument 5 \"observation_space\" "+\
                                            "must be a function.")

        # Setting the Ad-hoc components for the environment
        if issubclass(type(components),dict):
            if(not isinstance(components,dict)):
                raise ValueError("Components of environment must be a dictionary")

            self.components = self.copy_components(components)

        else:
            raise ValueError("argument 6 \"adhoc\" must be a "+\
                                str(dict)+" instance.")

    def copy_components(self,data):
        if (isinstance(data, list)):
            c = []
            for x in data:
                c.append(self.copy_components(x))
            return c

        elif (isinstance(data, dict)):
            c = {}
            for key in data.keys():
                c[key] = self.copy_components(data[key])
            return c

        elif (isinstance(data, int) or isinstance(data, float) or isinstance(data, complex) or isinstance(data, str) or isinstance(data, tuple)):
            return data

        else:
            try:
                if (hasattr(data, 'copy')):
                    return data.copy()
                else:
                    raise ValueError("Custom classes in \"components\" must have a copy method.")
                    
            except:
                raise NotImplementedError("Data type \""+str(type(data))+"\" not implemented.")
    
    def get_observation(self):
        return self.observation_space(self.copy())

    def step(self,action):
        # Execute one time step within the environment
        current_state = deepcopy(self.state)

        # 1. Simulating the action and getting the observation
        next_state, info = self.transition_function(action,self)
        observation = self.get_observation()
        self.episode += 1
        
        # 2. Calculating the reward
        # a. state transition based reward
        reward = self.reward_function(current_state, next_state)

        # b. action based reward
        reward += sum([info[key] if 'reward' in key else 0 for key in info])
        
        # 3. Verifying end condition
        done = self.state_set.is_final_state(next_state)
        
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.episode = 0
        self.close()
        if self.state_set.initial_state is not None and self.state_set.initial_components is not None:
            self.state = deepcopy(self.state_set.initial_state)
            self.components = self.copy_components(self.state_set.initial_components)
            return self.observation_space(self.copy())
        else:
            raise ValueError("the initial state from the state set is None.")

    def copy(self):
        copied_env = AdhocReasoningEnv(self.state_set,self.transition_function,\
            self.action_space,self.reward_function,self.observation_space,\
                                                        self.components)
        copied_env.viewer = self.viewer
        copied_env.state = deepcopy(self.state)
        copied_env.episode = self.episode
        copied_env.state_set.initial_state = \
            deepcopy(self.state_set.initial_state)
        return copied_env

    def render(self, mode='human'):
        # Render the environment to the screen
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
