from copy import deepcopy
import gym
from gym import spaces
from inspect import isfunction
from warnings import warn

class AdhocAgent:
    """ Adhoc Agent Class : Base class for all intelligent agents. Ideally, every environment should have at least one AdhocAgent subclass.
    To be implemented by the subclasses :
    *   copy() : Creates a copy of the agent. If not implemented, the default copy() is called.
    Example Usage :
    ```python
    class CustomAgent(AdhocAgent):
        def __init__(self,index,type,*args):
            super(CustomAgent,self).__init__(index,type)
            self.properties = args
        def copy():
            copied_agent = CustomAgent(self.index,self.type,self.properties)
            copied_agent.properties = deepcopy(self.properties) # Make sure it is not shallow copy
            copied_agent.smart_parameters = self.smart_parameters # This can be very large, so deepcopy not suggested.
            return copied_agent

    ```
    """
    def __init__(self,index,atype=None):
        """ Initializer for the AdhocAgent
        :param index: Unique identifier of the agent
        :param atype: Type of reasoning used by the agent. This must match with a reasoning method in src/reasoning
        """
        self.index = index
        self.type = atype
        
        self.next_action = None # Stores the next action of the agent
        self.target = None   # The target may or may not be used in all environments.
        self.smart_parameters = {}   # This is a suggested approach towards modelling Intelligent Agent knowledge.
    
    def copy(self):
        """
        :return: Creates a copy of the agent.
        """
        return AdhocAgent(self.index,self.type)

class StateSet:
    """StateSet : This is the class to model the initial state and end condition of environment. This is a mandatory part of any custom environment
                  It is used to specify the structure of our state [For ex : 2D grid ], end condition (a boolean function) and initial state and components.
                  It is used to reinitialise the state using env.reset()

    Example Usage :
     def end_condition(state):
        return False
     state_set = StateSet(spaces.Box(low=-1, high=np.inf, shape=shape, dtype=np.int64), end_condition)
     """
    def __init__(self,state_representation,end_condition):
        """Initializer for StateSet. Used to define the state formally in space.
        :param state_representation: Derived from gym.spaces.Space .
            For ex : space.Box(-1,np.inf,(10,10),dtype=np.int64)
        :param end_condition: Function which return True if given state is final state.
            For ex : def end_condition(state):
                        if(np.sum(state)>10):
                            return True
                        return False
        """
        if issubclass(type(state_representation),spaces.Space) or issubclass(state_representation,spaces.Space):
            self.state_representation = state_representation
        else:
            raise ValueError("argument \"state_representation\" must be a "+str(spaces.Space)+\
                                                " instance or inherit it.")
        
        # Defining the initial state
        self.initial_state = None # This can be any chosen form of representation, like array, list, dict etc.

        #Initial value of components. "components" are the generic version of specific, parts of enviorment, like players, tasks etc.
        self.initial_components = None #

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
    """AdhocReasoningEnv : Base class offered for all environments. It is the point of connection with OpenAI gym.
    To be implemented by subclass :
    * Render : It is the openAI gym function used to display the graphical interface. Render function is mandatory.
               Look at environments offered by the project to understand the process to create the rendering.
    * Copy : It is not mandatory but is highly suggested to ensure complete functionality of the base class. The default
            copy function will return an instance of the AdhocReasoningEnv Class.
    """
    colors = { \
        'red':          (255*1.0, 255*0.0, 255*0.0), \
        'darkred':      (255*0.5, 255*0.0, 255*0.0), \
        'green':        (255*0.0, 255*1.0, 255*0.0), \
        'darkgreen':    (255*0.0, 255*0.5, 255*0.0), \
        'blue':         (255*0.0, 255*0.0, 255*1.0), \
        'darkblue':     (255*0.0, 255*0.0, 255*0.5), \
        'cyan':         (255*0.0, 255*1.0, 255*1.0), \
        'darkcyan':     (255*0.0, 255*0.5, 255*0.5), \
        'magenta':      (255*1.0, 255*0.0, 255*1.0), \
        'darkmagenta':  (255*0.5, 255*0.0, 255*0.5), \
        'yellow':       (255*1.0, 255*1.0, 255*0.0), \
        'darkyellow':   (255*0.5, 255*0.5, 255*0.0), \
        'brown':        (255*0.0, 255*0.2, 255*0.2), \
        'white':        (255*1.0, 255*1.0, 255*1.0), \
        'lightgrey':    (255*0.8, 255*0.8, 255*0.8), \
        'darkgrey':     (255*0.15, 255*0.15, 255*0.15), \
        'black':        (255*0.0, 255*0.0, 255*0.0)
    }

    colors_percent = { \
        'red':          (1.0, 0.0, 0.0), \
        'darkred':      (0.5, 0.0, 0.0), \
        'green':        (0.0, 1.0, 0.0), \
        'darkgreen':    (0.0, 0.5, 0.0), \
        'blue':         (0.0, 0.0, 1.0), \
        'darkblue':     (0.0, 0.0, 0.5), \
        'cyan':         (0.0, 1.0, 1.0), \
        'darkcyan':     (0.0, 0.5, 0.5), \
        'magenta':      (1.0, 0.0, 1.0), \
        'darkmagenta':  (0.5, 0.0, 0.5), \
        'yellow':       (1.0, 1.0, 0.0), \
        'darkyellow':   (0.5, 0.5, 0.0), \
        'brown':        (0, 0.2, 0.2), \
        'white':        (1.0, 1.0, 1.0), \
        'lightgrey':    (0.8, 0.8, 0.8), \
        'darkgrey':     (0.4, 0.4, 0.4), \
        'black':        (0.0, 0.0, 0.0)
    }
    
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 50,
    }

    def __init__(self,state_set,transition_function,action_space,\
                    reward_function,observation_space,components):
        """Initialise the Ad-hoc Reasoning Environment
        :param state_set : dtype = StateSet , contains the state representation, initial state and end condition.
        :param transition_function : Takes action as input and returns next state and info variable. Info can be used to store step specific information.
                                     For example : Refer to levelforaging_transition in src/envs/LevelForagingEnv.py
        :param action_space : dtype : gym.spaces.Space , The action space for the environment.
        :param reward_function : Takes two successive states as input and returns reward for the step.
        :param observation_space : Takes the current CustomEnv class as input and returns observable form of the environment.
        :param components : dtype : dict() , Dictionary of all characteristic classes and parts of the environment.
        """

        super(AdhocReasoningEnv, self).__init__()
        # Graphical inteface.
        self.screen = None  
        self.render_mode = None
        self.renderer = None
        self.state = None   # The state can be any chosen form of representation. For ex : np array.
        self.episode = 0    # Number of episodes after env.reset()
        self.simulation = False
        # Setting the Markovian components for the environment
        # State
        if isinstance(state_set,StateSet):
            self.state_set = state_set
        else:
            raise ValueError("argument 1 \"state_set\" must be a "+\
                                            str(StateSet)+" instance.")

        # Step function
        if isfunction(transition_function):
            if transition_function.__code__.co_argcount != 2:
                raise ValueError("argument 2 \"transition_function\" "+\
                 "must be a function with 2 argument: the environment and"+\
                                        "the action of an agent (integer).")
            self.transition_function = transition_function
        else:
            raise ValueError("argument 2 \"transition_function\" "+\
                                                "must be a function.")

        # Action space
        if issubclass(type(action_space),spaces.Space):
            self.action_space = action_space
        else:
            raise ValueError("argument 3 \"action_space\" must be a "+\
                            str(spaces.Space)+" instance or inherit it.")

        # Reward function
        if isfunction(reward_function):
            self.reward_function = reward_function
        else:
            raise ValueError("argument 4 \"reward_function\" "+\
                                            "must be a function.")

        # Observation space
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
            raise ValueError("argument 6 \"components\" must be a "+\
                                str(dict)+" instance.")

    def copy_components(self,data):
        """ Generic copying of components of the environment

        :param data: The component to be copied
        :return: The copied component
        """

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
        
        elif data is None:
            return None

        else:
            try:
                if (hasattr(data, 'copy')):
                    return data.copy()
                else:
                    raise ValueError("Custom classes in \"components\" must have a copy method.")
                    
            except:
                raise NotImplementedError("Data type \""+str(type(data))+"\" not implemented.")
    
    def get_observation(self):
        """
        :return: The Observable form of the environment. Return type : AdhocReasoningEnv
        """
        return self.observation_space(self.copy())

    def step(self,action):
        """Execute one time step within the environment
        :param action: The action to be taken. dtype = int, float for discrete, continuous action space resp.
        :return: observation : dtype = AdhocReasoningEnv (Or DerivedClassEnv)
                 reward : dtype = int/float
                 done : dtype = bool
                 info : dtype = dict() . info['reward'] can be used for additional rewards corresponding to transition.

        How the env.step() works :
        Every time the env.step(action) is called, the env (which is an instance of subclass of AdhocReasoningEnv) will
        call the transition function with the given action. The env.state is updated accordingly. Now the observation is
        returned to the user to use for further reasoning regarding the future steps. This facilitates implementation of
        partial observability within the environment.
        """
        current_state = deepcopy(self.state)

        # 1. Simulating the action and getting the observation
        next_state, info = self.transition_function(action,self)
        if self.simulation:
            observation = self.copy()
        else:
            observation = self.get_observation()
        self.episode += 1
        
        # 2. Calculating the reward
        # a. state transition based reward
        reward = self.reward_function(current_state, next_state)

        # b. action based reward
        if info is not None:
            reward += sum([info[key] if 'reward' in key else 0 for key in info])
        
        # 3. Verifying end condition
        done = self.state_set.is_final_state(next_state)
        
        if self.renderer is not None and not self.simulation:
            self.renderer.render_step()
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.episode = 0
        if self.renderer is not None:
            self.renderer.reset()
            self.renderer.render_step()

        if self.state_set.initial_state is not None and self.state_set.initial_components is not None:
            self.state = deepcopy(self.state_set.initial_state)
            self.components = self.copy_components(self.state_set.initial_components)
            return self.observation_space(self.copy())

        else:
            raise ValueError("the initial state from the state set is None.")

    def copy(self):
        """ Copy the Environment
        Can (and should ) be over ridden to have better environment specific copying
        :return: Copied Instance of AdhocReasoningEnv
        """
        copied_env = AdhocReasoningEnv(self.state_set,self.transition_function,\
            self.action_space,self.reward_function,self.observation_space,\
                                                        self.components)
        copied_env.screen = self.screen
        copied_env.state = deepcopy(self.state)
        copied_env.episode = self.episode
        copied_env.state_set.initial_state = \
            deepcopy(self.state_set.initial_state)
        return copied_env

    def render(self, mode='human'):
        # Render the environment to the screen
        raise NotImplementedError

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
