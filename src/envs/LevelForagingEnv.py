from copy import deepcopy
from importlib import import_module
from gym import spaces
import numpy as np
import random as rd
import os

from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display=False):
    scenario, scenario_id = load_default_scenario_components(method,scenario_id)

    dim = scenario['dim']
    visibility = scenario['visibility']
    components = {'agents':scenario['agents'],'adhoc_agent_index':scenario['adhoc_agent_index'],'tasks':scenario['tasks']}
    env = LevelForagingEnv(shape=dim,components=components,visibility=visibility,display=display)
    return env, scenario_id

def load_default_scenario_components(method,scenario_id):
    if scenario_id >= 2:
        print('There is no default scenario with id '+str(scenario_id)+' for the LevelForaging problem. Setting scenario_id to 0.')
        scenario_id = 0

    default_scenarios_components = [
        {
        # Scenario 0: Simple Foraging Scenario
        'dim': (10,10),
        'visibility': 'partial',
        'agents' : [
            Agent(index='A',atype=method,position=(1,1),direction=1*np.pi/2,radius=0.25,angle=1,level=1.0), 
                ],
        'adhoc_agent_index' : 'A',
        'tasks' : [
            Task(index='0',position=(8,8),level=1.0),
            Task(index='1',position=(5,5),level=1.0),
            Task(index='2',position=(0,0),level=1.0),
            Task(index='3',position=(9,1),level=1.0),
            Task(index='4',position=(0,9),level=1.0)
                ]
        },
        {
        # Scenario 1: Level Foraging Scenario with Multiple Agents
        'dim': (10,10),
        'visibility': 'partial',
        'agents' : [
            Agent(index='A',atype=method,position=(1,1),direction=1*np.pi/2,radius=1,angle=1,level=1.0),
            Agent(index='B',atype='l1',position=(3,3),direction=np.pi,radius=0.25,angle=1,level=1), 
            Agent(index='C',atype='l2',position=(3,4),direction=np.pi,radius=0.5,angle=1,level=1) 
                ],
        'adhoc_agent_index' : 'A',
        'tasks' : [
            Task(index='0',position=(8,8),level=1.0),
            Task(index='1',position=(5,5),level=1.0),
            Task(index='2',position=(0,0),level=1.0),
            Task(index='3',position=(9,1),level=1.0),
            Task(index='4',position=(0,9),level=1.0)
                ]
        }]


    return default_scenarios_components[scenario_id], scenario_id

"""
    Support classes
"""
class Agent(AdhocAgent):
    """Agent : Main reasoning Component of the Environment. Derives from AdhocAgent Class
    """

    def __init__(self, index, atype, position, direction, radius, angle, level, estimation_method=None):
        super(Agent, self).__init__(index, atype)

        # agent parameters
        self.position = position
        self.direction = direction
        self.radius = radius
        self.angle = angle
        self.level = level

        self.smart_parameters['last_completed_task'] = None
        self.smart_parameters['choose_task_state'] = None
        self.smart_parameters['ntasks'] = None
        self.smart_parameters['estimation_method'] = estimation_method

    def copy(self):
        # 1. Initialising the agent
        copy_agent = Agent(self.index, self.type, deepcopy(self.position), \
                           self.direction, self.radius, self.angle, self.level)

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters

        return copy_agent

    def set_parameters(self, parameters):
        self.radius = parameters[0]
        self.angle = parameters[1]
        self.level = parameters[2]

    def get_parameters(self):
        return np.array([self.radius,self.angle,self.level])

    def show(self):
        print(self.index, self.type, ':', self.position, self.direction, self.radius, self.angle, self.level)


class Task():
    """Task : These are parts of the 'components' of the environemnt.
    """

    def __init__(self, index, position, level):
        # task parameters
        self.index = index
        self.position = position
        self.level = level

        # task simulation parameters
        self.completed = False
        self.trying = []

    def copy(self):
        # 1. Initialising the copy task
        copy_task = Task(self.index, self.position, self.level)

        # 2. Copying the parameters
        copy_task.completed = self.completed
        copy_task.trying = [a for a in self.trying]

        return copy_task


"""
    Customising the Level-Foraging Env
"""


def end_condition(state):
    return sum(sum(state.state == np.inf)) == 0


def who_see(env, position):
    who = []
    for a in env.components['agents']:
        if a.direction is not None:
            direction = a.direction
        else:
            # TODO : Correct this
            direction = env.action_space.sample()

        if a.radius is not None:
            radius = np.sqrt(env.state.shape[0] ** 2 + env.state.shape[1] ** 2) * a.radius
        else:
            radius = np.sqrt(env.state.shape[0] ** 2 + env.state.shape[1] ** 2) * rd.uniform(0, 1)

        if a.radius is not None:
            angle = 2 * np.pi * a.angle
        else:
            angle = 2 * np.pi * rd.uniform(0, 1)

        if is_visible(position, a.position, direction, radius, angle):
            who.append(a)
    return who


def there_is_task(env, position, direction):
    # 1. Calculating the task position
    if direction == np.pi / 2:
        pos = (position[0], position[1] + 1)
    elif direction == 3 * np.pi / 2:
        pos = (position[0], position[1] - 1)
    elif direction == np.pi:
        pos = (position[0] - 1, position[1])
    elif direction == 0:
        pos = (position[0] + 1, position[1])
    else:
        pos = None

    # 2. If there is a task, return it, else None
    for task in env.components['tasks']:
        if not task.completed:
            if pos == task.position:
                return task
    return None


def new_position_given_action(state, pos, action):
    # 1. Calculating the new position
    if action == 2:  # N
        new_pos = (pos[0], pos[1] + 1) if pos[1] + 1 < state.shape[0] \
            else (pos[0], pos[1])
    elif action == 3:  # S
        new_pos = (pos[0], pos[1] - 1) if pos[1] - 1 >= 0 \
            else (pos[0], pos[1])
    elif action == 1:  # W
        new_pos = (pos[0] - 1, pos[1]) if pos[0] - 1 >= 0 \
            else (pos[0], pos[1])
    elif action == 0:  # E
        new_pos = (pos[0] + 1, pos[1]) if pos[0] + 1 < state.shape[1] \
            else (pos[0], pos[1])
    else:
        new_pos = (pos[0], pos[1])

    # 2. Verifying if it is empty and in the map boundaries
    if state[new_pos[0], new_pos[1]] == 0:
        return new_pos
    else:
        return pos


# This method returns the visible tasks positions
def get_visible_components(state, agent):
    # 1. Defining the agent vision parameters
    direction = agent.direction
    radius = np.sqrt(state.shape[0] ** 2 + state.shape[1] ** 2) * agent.radius
    angle = 2 * np.pi * agent.angle

    agents, tasks = [], []

    # 2. Looking for tasks
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            if (x, y) != agent.position:
                if is_visible([x, y], agent.position, direction, radius, angle):
                    if state[(x, y)] == 1:
                        agents.append([x, y])
                    elif state[(x, y)] == np.inf:
                        tasks.append([x, y])

    # 3. Returning the result
    return {'agents':agents, 'tasks':tasks}


# This method returns the distance between an object and a viewer
def distance(obj, viewer):
    return np.sqrt((obj[0] - viewer[0]) ** 2 + (obj[1] - viewer[1]) ** 2)


# This method returns true if an object is in the viewer vision angle
def angle_of_gradient(obj, viewer, direction):
    xt = (obj[0] - viewer[0])
    yt = (obj[1] - viewer[1])

    x = np.cos(direction) * xt + np.sin(direction) * yt
    y = -np.sin(direction) * xt + np.cos(direction) * yt
    if (y == 0 and x == 0):
        return 0
    return np.arctan2(y, x)


# This method returns True if a position is visible, else False
def is_visible(obj, viewer, direction, radius, angle):
    # 1. Checking visibility
    if distance(obj, viewer) <= radius \
            and -angle / 2 <= angle_of_gradient(obj, viewer, direction) <= angle / 2:
        return True
    else:
        return False


def update(env):
    # 1. Cleaning the map components (agents and tasks)
    env.state = np.zeros_like(env.state)

    # 2. Updating its components
    for agent in env.components['agents']:
        x, y = agent.position[0], agent.position[1]
        env.state[x, y] = 1

    for task in env.components['tasks']:
        x, y = task.position[0], task.position[1]

        if not task.completed:
            env.state[x, y] = np.inf

    return env


def do_action(env):
    # 1. Position and direction
    # a. defining the agents new position and direction
    state, components = env.state, env.components
    positions, directions = {}, {}
    action2direction = {
        0: 0,  # East
        1: np.pi,  # West
        2: np.pi / 2,  # North
        3: 3 * np.pi / 2}  # South
    info = {'action reward': 0, 'just_finished_tasks': []}

    for agent in components['agents']:
        if agent.next_action != 4 and agent.next_action is not None:
            positions[agent.index] = new_position_given_action(state, agent.position, agent.next_action)
            directions[agent.index] = action2direction[agent.next_action]

        else:
            positions[agent.index] = agent.position
            directions[agent.index] = agent.direction

    # b. analysing position conflicts
    for i in range(len(components['agents'])):
        for j in range(i + 1, len(components['agents'])):
            if positions[components['agents'][i].index] == \
                    positions[components['agents'][j].index]:
                if rd.uniform(0, 1) < 0.5:
                    positions[components['agents'][i].index] = \
                        components['agents'][i].position
                else:
                    positions[components['agents'][j].index] = \
                        components['agents'][j].position

    # c. updating the simulation agents position
    for i in range(len(components['agents'])):
        components['agents'][i].position = positions[components['agents'][i].index]
        components['agents'][i].direction = directions[components['agents'][i].index]

    # 2. Tasks 
    # a. verifying the tasks to be completed
    for agent in components['agents']:
        if agent.next_action == 4:
            task = there_is_task(env, agent.position, agent.direction)
            if task is not None:
                if agent.level is not None:
                    task.trying.append(agent.level)
                else:
                    task.trying.append(rd.uniform(0, 1))
        else:
            agent.smart_parameters['last_completed_task'] = None

    # b. calculating the reward
    for task in components['tasks']:
        #print(task.completed)
        if not task.completed:
            if sum([level for level in task.trying]) >= task.level:
                #info['action reward'] += 1
                task.completed = True
                if task not in info['just_finished_tasks']:
                    info['just_finished_tasks'].append(task)

        for ag in who_see(env, task.position):
            if task.completed and (ag.target == task.position or ag.target == task.index):
                if not env.simulation:
                    ag.smart_parameters['last_completed_task'] = task
                    ag.smart_parameters['choose_task_state'] = env.copy()
                ag.target = None

    # c. resetting the task trying
    for task in components['tasks']:
        task.trying = []

    if not env.simulation:
        for ag in env.components['agents']:
            ag.smart_parameters['ntasks'] -= len(info['just_finished_tasks'])
    next_state = update(env)

    return next_state, info


def get_target_non_adhoc_agent(agent, real_env):
    # changing the perspective
    copied_env = real_env.copy()

    # generating the observable scenario
    observable_env = copied_env.observation_space(copied_env)

    # planning the action from agent i perspective
    if agent.type is not None:
        
        try:
            module = import_module('src.reasoning.levelbased.'+agent.type)
        except:
            module = import_module('src.reasoning.'+agent.type)
        
        planning_method = getattr(module, agent.type + '_planning')

        agent.next_action, agent.target = \
            planning_method(observable_env, agent)
    else:
        agent.next_action, agent.target = \
            real_env.action_space.sample(), None

    # retuning the results
    return agent.target


def levelforaging_transition(action, real_env):
    # agent planning
    adhoc_agent_index = real_env.components['agents'].index(real_env.get_adhoc_agent())

    for i in range(len(real_env.components['agents'])):
        if i != adhoc_agent_index:
            # changing the perspective
            copied_env = real_env.copy()
            copied_env.components['adhoc_agent_index'] = copied_env.components['agents'][i].index

            # generating the observable scenario
            obsavable_env = copied_env.observation_space(copied_env)

            # planning the action from agent i perspective
            if real_env.components['agents'][i].type is not None:
                try:
                    module = import_module('src.reasoning.levelbased.'+real_env.components['agents'][i].type)
                except:
                    module = import_module('src.reasoning.'+real_env.components['agents'][i].type)
                planning_method = getattr(module, real_env.components['agents'][i].type + '_planning')

                real_env.components['agents'][i].next_action, real_env.components['agents'][i].target = \
                    planning_method(obsavable_env, real_env.components['agents'][i])
            else:
                real_env.components['agents'][i].next_action, real_env.components['agents'][i].target = \
                    real_env.action_space.sample(), None

        else:
            real_env.components['agents'][i].next_action = action
            real_env.components['agents'][i].target = real_env.components['agents'][i].target

    # environment step
    next_state, info = do_action(real_env)

    # retuning the results
    return next_state, info


# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return sum(sum(state == np.inf)) - (sum(sum(next_state.state == np.inf)))
    #return 0


# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    if copied_env.simulation:
        return copied_env
        
    agent = copied_env.get_adhoc_agent()
    if agent.radius is not None:
        radius = np.sqrt(copied_env.state.shape[0] ** 2 + copied_env.state.shape[1] ** 2) * agent.radius
    else:
        radius = np.sqrt(copied_env.state.shape[0] ** 2 + copied_env.state.shape[1] ** 2) * rd.uniform(0, 1)

    if agent.radius is not None:
        angle = 2 * np.pi * agent.angle
    else:
        angle = 2 * np.pi * rd.uniform(0, 1)

    if agent is not None:
        # 1. Removing the invisible agents and tasks from environment
        invisible_agents = []
        for i in range(len(copied_env.components['agents'])):
            if copied_env.components['agents'].index != agent.index:
                if not is_visible(copied_env.components['agents'][i].position,
                                agent.position, agent.direction, radius, angle) and \
                        copied_env.components['agents'][i] != agent:
                    invisible_agents.append(i)

        for index in sorted(invisible_agents, reverse=True):
            copied_env.components['agents'].pop(index)

        invisible_tasks = []
        for i in range(len(copied_env.components['tasks'])):
            if not is_visible(copied_env.components['tasks'][i].position,
                              agent.position, agent.direction, radius, angle) or \
                    copied_env.components['tasks'][i].completed:
                invisible_tasks.append(i)

        for index in sorted(invisible_tasks, reverse=True):
            copied_env.components['tasks'].pop(index)

        # 2. Building the observable environment
        copied_env.state = np.zeros(copied_env.state.shape)

        # a. setting the visible components
        # - main agent
        pos = agent.position
        copied_env.state[pos[0], pos[1]] = 1

        # - teammates
        for a in copied_env.components['agents']:
            pos = a.position
            copied_env.state[pos[0], pos[1]] = 1

        # - tasks
        for t in copied_env.components['tasks']:
            pos = t.position
            copied_env.state[pos[0], pos[1]] = np.inf

        # b. cleaning agents information
        if copied_env.visibility == 'partial':
            for i in range(len(copied_env.components['agents'])):
                if copied_env.components['agents'][i] != agent:
                    copied_env.components['agents'][i].radius = None
                    copied_env.components['agents'][i].angle = None
                    copied_env.components['agents'][i].level = None
                    copied_env.components['agents'][i].target = None
                    copied_env.components['agents'][i].type = None

        copied_env.episode += 1
        copied_env = update(copied_env)
        return copied_env
    else:
        raise IOError(agent, 'is an invalid agent.')


"""
    Level-Foraging Environments 
"""


class LevelForagingEnv(AdhocReasoningEnv):

    action_dict = {
        0: 'East',
        1: 'West',
        2: 'North',
        3: 'South',
        4: 'Load'
    }

    agents_color = {
        'mcts': 'red',
        'pomcp': 'yellow',
        'ibpomcp':'blue',
        'rhopomcp':'cyan',
        'l1': 'darkred',
        'l2': 'darkgreen',
        'l3': 'darkcyan',
    }

    def __init__(self, shape, components, visibility='full',display=False):
        ###
        # Env Settings
        ###
        self.visibility = visibility

        state_set = StateSet(spaces.Box( \
            low=-1, high=np.inf, shape=shape, dtype=np.int64), end_condition)
        transition_function = levelforaging_transition
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(LevelForagingEnv, self).__init__(state_set, \
                                               transition_function, action_space, reward_function, \
                                               observation_space, components)

        # Setting the inital state
        self.state_set.initial_state = np.zeros(shape)
        for element in components:
            if element == 'agents':
                for ag in components[element]:
                    self.state_set.initial_state[ag.position[0], ag.position[1]] = 1

            if element == 'tasks':
                for tk in components[element]:
                    self.state_set.initial_state[tk.position[0], tk.position[1]] = np.inf

            if element == 'obstacle':
                for ob in components[element]:
                    self.state_set.initial_state[ob.position[0], ob.position[1]] = -1
                
        # Updating agent knowledge about tasks
        for i in range(len(components['agents'])):
            self.components['agents'][i].smart_parameters['ntasks'] = len(components['tasks'])

        # Setting the initial components
        agent = self.get_adhoc_agent()
        self.state_set.initial_components = self.copy_components(components)
        self.empty_position = self.init_out_range_position(agent)

        ###
        # Setting graphical interface
        ###
        self.screen = None
        self.display = display
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

    def show_state(self):
        for y in reversed(range(self.state.shape[1])):
            for x in range(self.state.shape[0]):
                print(self.state[x,y],end=' ')
            print()

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.levelbased.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = LevelForagingEnv(self.state.shape, components, self.visibility)
        copied_env.simulation = self.simulation
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.renderer = self.renderer

        # Setting the initial state
        copied_env.state = np.array(
            [np.array([self.state[x, y] for y in range(self.state.shape[1])]) for x in range(self.state.shape[0])])
        copied_env.episode = self.episode
        copied_env.empty_position = [pos for pos in self.empty_position]
        copied_env.state_set.initial_state = np.zeros(copied_env.state.shape)
        for x in range(self.state_set.initial_state.shape[0]):
            for y in range(self.state_set.initial_state.shape[1]):
                copied_env.state_set.initial_state[x, y] = self.state_set.initial_state[x, y]
        return copied_env

    def get_actions_list(self):
        return [0,1,2,3,4]

    def get_adhoc_agent(self):
        for agent in self.components['agents']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        raise IndexError("Ad-hoc Index is not in Agents Index Set.")

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        return [self,1]
        
    def state_is_equal(self, state):
        for x in range(self.state.shape[0]):
            for y in range(self.state.shape[1]):
                if self.state[x, y] != state.state[x, y]:
                    return False
        return True

    def observation_is_equal(self, obs):
        cur_visibility = get_visible_components(self.state,self.get_adhoc_agent())
        obs_visibility = get_visible_components(obs.state,obs.get_adhoc_agent())
        return (cur_visibility['agents'] == obs_visibility['agents']) and (cur_visibility['tasks'] == obs_visibility['tasks'])

    def init_out_range_position(self, agent):
        empty_spaces = []

        dim_w, dim_h = self.state_set.initial_state.shape
        direction = agent.direction
        radius = np.sqrt(dim_w ** 2 + dim_h ** 2) * agent.radius
        angle = 2 * np.pi * agent.angle

        for x in range(dim_w):
            for y in range(dim_h):
                if not is_visible((x, y), agent.position, direction, radius, angle):
                    empty_spaces.append((x, y))
        return empty_spaces

    def get_out_range_position(self, agent):
        empty_spaces = []

        dim_w, dim_h = self.state.shape
        direction = agent.direction
        radius = np.sqrt(dim_w ** 2 + dim_h ** 2) * agent.radius
        angle = 2 * np.pi * agent.angle

        for x in range(dim_w):
            for y in range(dim_h):
                if not is_visible((x, y), agent.position, direction, radius, angle):
                    empty_spaces.append((x, y))
        return empty_spaces

    def sample_state(self, agent):
        # 1. Defining the base simulation
        u_env = self.copy()

        # - setting environment components
        dim_w, dim_h = self.state_set.initial_state.shape
        direction = agent.direction
        radius = np.sqrt(dim_w ** 2 + dim_h ** 2) * agent.radius
        angle = 2 * np.pi * agent.angle

        # - setting tasks
        for i in range(agent.smart_parameters['ntasks']):
            if len(u_env.empty_position) == 0:
                u_env.empty_position = self.get_out_range_position(agent)
                if len(u_env.empty_position) == 0:
                    break

            pos = rd.choice(u_env.empty_position)
            while is_visible(pos, agent.position, direction, radius, angle):
                u_env.empty_position.remove(pos)
                pos = rd.choice(u_env.empty_position)
                if len(u_env.empty_position) == 0:
                    u_env.empty_position = self.get_out_range_position(agent)
                
            u_env.state[pos[0], pos[1]] = np.inf
            u_env.components['tasks'].append(Task('S',pos,rd.uniform(0,1)))
            u_env.empty_position.remove(pos)

        return u_env

    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def get_target(self, agent_index, new_type=None, new_parameter=None):
        # changing the perspective
        copied_env = self.copy()
        copied_env.components['adhoc_agent_index'] = agent_index

        # generating the observable scenario
        adhoc_agent = copied_env.get_adhoc_agent()
        adhoc_agent.type = new_type
        adhoc_agent.set_parameters(new_parameter)
        adhoc_agent.target = None

        obsavable_env = copied_env.observation_space(copied_env)

        obsavable_env.components['adhoc_agent_index'] = agent_index
        adhoc_agent = obsavable_env.get_adhoc_agent()
        adhoc_agent.type = new_type
        adhoc_agent.set_parameters(new_parameter)
        adhoc_agent.target = None

        # planning the action from agent i perspective
        try:
            module = import_module('src.reasoning.levelbased.'+new_type)
        except:
            module = import_module('src.reasoning.'+new_type)

        planning_method = getattr(module,  new_type + '_planning')
        _, target = \
            planning_method(obsavable_env, adhoc_agent)

        # retuning the results
        for task in self.components['tasks']:
            if task.position == target:
                return task
        return None

    def render(self):
        return self.renderer.get_renders()
        
    def _render(self, mode="human"):
        ##
        # Standard Imports
        ##
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
            from gym.error import DependencyNotInstalled
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        self.screen_width, self.screen_height = 800, 800
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        ##
        # Drawing
        ##
        if self.state is None:
            return None

        dim = self.state.shape
        # background
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill(self.colors['white'])
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # grid
        grid_width, grid_height = 700, 700
        self.grid_surf = pygame.Surface((grid_width, grid_height))
        self.grid_surf.fill(self.colors['white'])

        for column in range(-1,dim[1]):
            pygame.draw.line(self.grid_surf,self.colors['black'],
                                (0*grid_width,(column+1)*(grid_height/dim[1])),
                                (1*grid_width,(column+1)*(grid_height/dim[1])),
                                int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
        for row in range(-1,dim[0]):
            pygame.draw.line(self.grid_surf,self.colors['black'],
                                ((row+1)*(grid_width/dim[0]),0*grid_height),
                                ((row+1)*(grid_width/dim[0]),1*grid_height),
                                int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))

        # agents
        self.components_surf = pygame.Surface((grid_width, grid_width))
        self.components_surf = self.components_surf.convert_alpha()
        self.components_surf.fill((self.colors['white'][0],self.colors['white'][1],self.colors['white'][2],0))
        def my_rotation(ox,oy,px,py,angle):
            angle = angle
            qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
            qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
            return int(qx),int(qy)

        for agent in self.components['agents']:
            direction = agent.direction - np.pi/2
            ox = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            oy = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
            #arms
            w = int(0.85*(grid_width/dim[0]))
            h = int(0.25*(grid_height/dim[1]))
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))

            arms = pygame.Surface((w , h))  
            arms.set_colorkey(self.colors['white'])  
            arms.fill(self.colors['black'])  
            arms = pygame.transform.rotate(arms, np.rad2deg(direction))
            arms_rec = arms.get_rect(center=(ox,oy))
            self.components_surf.blit(arms,arms_rec)
            
            #body
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
            r = int(0.35*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
            r = int(0.3*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors[self.agents_color[agent.type]])
            #eyes
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.4*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.15*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.6*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.15*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.4*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['white'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.6*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.8*(grid_height/dim[1]))
            r = int(0.1*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['white'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.4*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.85*(grid_height/dim[1]))
            r = int(0.07*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.6*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.85*(grid_height/dim[1]))
            r = int(0.07*np.sqrt((grid_width/dim[0])*(grid_height/dim[1])))
            x, y = my_rotation(ox,oy,x,y,direction)
            gfxdraw.filled_circle(self.components_surf,x,y,r,self.colors['black'])
            # index
            agent_idx = str(agent.index)
            myfont = pygame.font.SysFont("Ariel", int(0.6*np.sqrt((grid_width/dim[0])*(grid_height/dim[1]))))
            label = myfont.render(agent_idx, True, self.colors['black'])
            x = int(agent.position[0]*(grid_width/dim[0]) + 0.35*(grid_width/dim[0]))
            y = int(agent.position[1]*(grid_height/dim[1]) + 0.3*(grid_height/dim[1]))
            label =  pygame.transform.flip(label, False, True)
            self.components_surf.blit(label, (x,y))

        # box
        adhoc_agent = self.get_adhoc_agent()
        for task in self.components['tasks']:
            if not task.completed:
                rx, ry = task.position[0]*(grid_width/dim[0]),task.position[1]*(grid_height/dim[1])

                task_ret = pygame.Rect((rx+int(0.0*grid_width/dim[0]),ry+int(0.0*grid_height/dim[1])),\
                    (int(1*grid_width/dim[0]),int(1*grid_height/dim[1])))
                task_img = pygame.image.load(os.path.abspath("./imgs/levelbased/task_box.png"))
                task_img = pygame.transform.flip(task_img,False,True)
                task_img = pygame.transform.scale(task_img, task_ret.size)
                task_img = task_img.convert()

                
                dim_w, dim_h = self.state_set.initial_state.shape
                direction = adhoc_agent.direction
                radius = np.sqrt(dim_w ** 2 + dim_h ** 2) * adhoc_agent.radius
                angle = 2 * np.pi * adhoc_agent.angle
                if is_visible(task.position,adhoc_agent.position,direction,radius,angle):
                    self.components_surf.blit(task_img,task_ret)
                else:
                    self.grid_surf.blit(task_img,task_ret)

        ##
        # Text
        ##
        act = self.action_dict[adhoc_agent.next_action] \
            if adhoc_agent.next_action is not None else None
        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Episode "+str(self.episode) + \
            " | Action: "+str(act), True, self.colors['black'])
        self.screen.blit(label, (10, 10))
        
        # fog
        self.fog_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        self.fog_surf = self.fog_surf.convert_alpha()
        self.fog_surf.fill((self.colors['darkgrey'][0],self.colors['darkgrey'][1],self.colors['darkgrey'][2],100))
        self.fog_surf = pygame.transform.flip(self.fog_surf, False, True)

        # vision
        x = int(adhoc_agent.position[0]*(grid_width/dim[0]) + 0.5*(grid_width/dim[0]))
        y = int(adhoc_agent.position[1]*(grid_height/dim[1]) + 0.5*(grid_height/dim[1]))
        r = int(adhoc_agent.radius*np.sqrt((grid_width)**2+(grid_height)**2))
        self.vision_surf = pygame.Surface((grid_width, grid_height), pygame.SRCALPHA, 32)
        self.vision_surf = self.vision_surf.convert_alpha()
        gfxdraw.pie(self.vision_surf,x,y,r,
            int(np.rad2deg(adhoc_agent.direction-(np.pi*adhoc_agent.angle))),
            int(np.rad2deg(adhoc_agent.direction+(np.pi*adhoc_agent.angle))),
            (self.colors['black'][0],self.colors['black'][1],self.colors['black'][2],200))
        
        start_angle = adhoc_agent.direction-(np.pi*adhoc_agent.angle)
        stop_angle = adhoc_agent.direction+(np.pi*adhoc_agent.angle)
        theta = start_angle
        while theta <= stop_angle:
            pygame.draw.line(self.vision_surf,
                (self.colors['white'][0],self.colors['white'][1],self.colors['white'][2],100),
                    (x,y), (x+r*np.cos(theta),y+r*np.sin(theta)),10)
            theta += (stop_angle-start_angle)/100

        self.vision_surf = pygame.transform.flip(self.vision_surf, False, True)

        ##
        # Displaying
        ##
        self.grid_surf = pygame.transform.flip(self.grid_surf, False, True)
        self.components_surf = pygame.transform.flip(self.components_surf, False, True)
        self.screen.blit(self.grid_surf, (50, 50))
        self.screen.blit(self.fog_surf, (50, 50))
        self.screen.blit(self.vision_surf, (50, 50))
        self.screen.blit(self.components_surf, (50, 50))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )