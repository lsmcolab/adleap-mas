from datetime import datetime
import gym
from gym import error, spaces
import numpy as np
import random as rd

from .AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Rendering 
"""

"""
    Ad-hoc 
"""


class Agent(AdhocAgent):
    """Agent : Main reasoning Component of the Environment. Derives from AdhocAgent Class
    """

    def __init__(self, index, atype, position, direction, radius, angle, level):
        super(Agent, self).__init__(index, atype)

        # agent parameters
        self.position = position
        self.direction = direction
        self.radius = radius
        self.angle = angle
        self.level = level

    def copy(self):
        # 1. Initialising the agent
        copy_agent = Agent(self.index, self.type, self.position, \
                           self.direction, self.radius, self.angle, self.level)

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters

        return copy_agent

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
    return (sum(sum(state == np.inf)) == 0)


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
def get_visible_agents_and_tasks(state, agent, components):
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
                    if (x, y) == 1:
                        agents.append([x, y])
                    elif (x, y) == np.inf:
                        tasks.append([x, y])

    # 3. Returning the result
    return agents, tasks


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
    for x in range(env.state.shape[0]):
        for y in range(env.state.shape[1]):
            if env.state[x, y] > 0:
                env.state[x, y] = 0

    # 2. Updating its components
    for agent in env.components['agents']:
        x, y = agent.position[0], agent.position[1]
        env.state[x, y] = 1

    for task in env.components['tasks']:
        x, y = task.position[0], task.position[1]

        if not task.completed:
            env.state[x, y] = np.inf

    return env.state


def do_action(env):
    # 1. Position and direction
    # a. defining the agents new position and direction
    just_finished_tasks = []
    state, components = env.state, env.components
    positions, directions = {}, {}
    action2direction = {
        0: 0,  # East
        1: np.pi,  # West
        2: np.pi / 2,  # North
        3: 3 * np.pi / 2}  # South
    info = {'action reward': 0, 'just_finished_tasks': []}

    for agent in components['agents']:
        if agent.next_action != 4:
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
    for agent in components['agents']:
        agent.position = positions[agent.index]
        agent.direction = directions[agent.index]

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

    # b. calculating the reward
    for task in components['tasks']:
        #print(task.completed)
        if task.completed:
            continue
        if sum([level for level in task.trying]) >= task.level:
            # info['action reward'] += 1
            task.completed = True
            if task not in just_finished_tasks:
                just_finished_tasks.append(task)


        for ag in who_see(env, task.position):

            if task.completed and ag.target == task.position:
                if not env.simulation:
                    ag.smart_parameters['last_completed_task'] = task
                    ag.smart_parameters['choose_task_state'] = env.copy()
                ag.target = None

    # c. resetting the task trying
    for task in components['tasks']:
        task.trying = []

    next_state = update(env)
    info['just_finished_tasks'] = just_finished_tasks

    return next_state, info


def get_target_non_adhoc_agent(agent, real_env):
    # agent planning
    adhoc_agent_index = real_env.components['agents'].index(real_env.get_adhoc_agent())

    # changing the perspective
    copied_env = real_env.copy()

    # generating the observable scenario
    observable_env = copied_env.observation_space(copied_env)

    # planning the action from agent i perspective
    if agent.type is not None:
        module = __import__(agent.type)
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
                module = __import__(real_env.components['agents'][i].type)
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
    # return sum(sum(state == np.inf)) - (sum(sum(next_state == np.inf)))
    return 0


# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
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
        return copied_env
    else:
        raise IOError(agent, 'is an invalid agent.')


"""
    Level-Foraging Environments 
"""


class LevelForagingEnv(AdhocReasoningEnv):
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

    action_dict = {
        0: 'East',
        1: 'West',
        2: 'North',
        3: 'South',
        4: 'Load'
    }

    def __init__(self, shape, components, visibility='full',display=False):
        self.viewer = None
        self.visibility = visibility
        self.display = display
        # Defining the Ad-hoc Reasoning Env parameters
        state_set = StateSet(spaces.Box( \
            low=-1, high=np.inf, shape=shape, dtype=np.int64), end_condition)
        transition_function = levelforaging_transition
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        # Initialising the Adhoc Reasoning Env
        super(LevelForagingEnv, self).__init__(state_set, \
                                               transition_function, action_space, reward_function, \
                                               observation_space, components)
        self.agents_color = {}

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

        # Setting the initial components
        self.state_set.initial_components = self.copy_components(components)

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = LevelForagingEnv(self.state.shape, components, self.visibility)
        copied_env.simulation = self.simulation
        copied_env.viewer = self.viewer
        copied_env.state = np.array(
            [np.array([self.state[x, y] for y in range(self.state.shape[1])]) for x in range(self.state.shape[0])])
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state_set.initial_state = np.zeros(copied_env.state.shape)
        for x in range(self.state_set.initial_state.shape[0]):
            for y in range(self.state_set.initial_state.shape[1]):
                copied_env.state_set.initial_state[x, y] = self.state_set.initial_state[x, y]
        return copied_env

    def get_adhoc_agent(self):
        for agent in self.components['agents']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        return None

    def state_is_equal(self, state):
        for x in range(self.state.shape[0]):
            for y in range(self.state.shape[1]):
                if self.state[x, y] != state[x, y]:
                    return False
        return True

    def observation_is_equal(self, obs):
        observable_env = self.observation_space(self.copy())
        for x in range(observable_env.state.shape[0]):
            for y in range(observable_env.state.shape[1]):
                if observable_env.state[x, y] != obs.state[x, y]:
                    return False
        return True

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

    def sample_state(self, agent, sample_p=0.1, n_sample=10):
        # 1. Defining the base simulation
        u_env = self.copy()

        # - getting empty space out of range
        empty_position = u_env.get_out_range_position(agent)

        # - setting environment components
        count = 0
        while len(empty_position) > 0 and count < n_sample:
            # - setting teammates
            if rd.uniform(0, 1) < sample_p:
                pos = rd.sample(empty_position, 1)[0]
                u_env.state[pos[0], pos[1]] = 1
                empty_position.remove(pos)
            count += 1

            # - setting tasks
            if rd.uniform(0, 1) < sample_p:
                pos = rd.sample(empty_position, 1)[0]
                u_env.state[pos[0], pos[1]] = np.inf
                empty_position.remove(pos)
            count += 1

        return u_env

    def render(self, mode='human'):
        if not self.display:
            return

        try:
            global rendering
            from gym.envs.classic_control import rendering

        except ImportError as e:
            raise ImportError('''
            Cannot import rendering
            ''')
        try:
            global pyglet
            import pyglet
        except ImportError as e:
            raise ImportError('''
            Cannot import pyglet.
            HINT: you can install pyglet directly via 'pip install pyglet'.
            But if you really just want to install all Gym dependencies and not have to think about it,
            'pip install -e .[all]' or 'pip install gym[all]' will do it.
            ''')

        try:
            global glBegin,glEnd,GL_QUADS,GL_POLYGON,GL_TRIANGLES,glVertex3f
            from pyglet.gl import glBegin, glEnd, GL_QUADS, GL_POLYGON, GL_TRIANGLES, glVertex3f
        except ImportError as e:
            raise ImportError('''
            Error occurred while running `from pyglet.gl import *`
            HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
            If you're running on a server, you may need a virtual frame buffer; something like this should work:
            'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
            ''')
        global FilledPolygonCv4
        class FilledPolygonCv4(rendering.Geom):
            def __init__(self, v):
                rendering.Geom.__init__(self)
                self.v = v

            def render1(self):
                if len(self.v) == 4:
                    glBegin(GL_QUADS)
                elif len(self.v) > 4:
                    glBegin(GL_POLYGON)
                else:
                    glBegin(GL_TRIANGLES)
                for p in self.v:
                    glVertex3f(p[0], p[1], 0)  # draw each vertex
                glEnd()

            def set_color(self, r, g, b, a):
                self._color.vec4 = (r, g, b, a)


        global DrawText
        class DrawText(rendering.Geom):
            def __init__(self, label: pyglet.text.Label):
                rendering.Geom.__init__(self)
                self.label = label

            def render1(self):
                self.label.draw()



        # Render the environment to the screen
        adhoc_agent_index = self.components['agents'].index(self.get_adhoc_agent())
        if self.state is not None:
            if self.viewer is None:

                self.screen_width, self.screen_height, self.pad = 800, 800, 0.1
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
                self.draw_scale = (self.screen_width - (self.pad * self.screen_width)) / self.state.shape[0] \
                    if self.state.shape[0] > self.state.shape[1] else (self.screen_height - (
                        self.pad * self.screen_height)) / self.state.shape[1]
                self.draw_start_x = self.screen_width / 2 - (self.draw_scale * self.state.shape[0]) / 2
                self.draw_start_y = self.screen_height / 2 - (self.draw_scale * self.state.shape[1]) / 2

                # Drawing the environment
                self.drawn_agents = self.draw_agents()
                self.drawn_tasks, self.drawn_tasks_shift = self.draw_tasks(type_='figure', fname='imgs/task_box.png')

                for i in range(len(self.components['agents'])):
                    if self.components['agents'][adhoc_agent_index].index == self.components['agents'][i].index:
                        self.draw_fog()
                        self.draw_vision(self.components['agents'][i], self.drawn_agents[i])

                self.draw_grid()
                if 'obstacles' in self.components:
                    self.draw_obstacles()

            for i in range(len(self.components['agents'])):
                x, y = self.components['agents'][i].position[0], self.components['agents'][i].position[1]
                rotate = self.components['agents'][i].direction - np.pi / 2
                self.drawn_agents[i].set_rotation(rotate)
                self.drawn_agents[i].set_translation(
                    self.draw_start_x + (x + self.drawn_tasks_shift) * self.draw_scale,
                    (y + self.drawn_tasks_shift) * self.draw_scale + self.draw_start_y
                )

            for i in range(len(self.components['tasks'])):
                if not self.components['tasks'][i].completed:
                    x, y = self.components['tasks'][i].position[0], self.components['tasks'][i].position[1]
                    self.drawn_tasks[i].set_translation(
                        self.draw_start_x + (x + self.drawn_tasks_shift) * self.draw_scale,
                        (y + self.drawn_tasks_shift) * self.draw_scale + self.draw_start_y
                    )
                else:
                    self.drawn_tasks[i].set_scale(0.0, 0.0)

            self.draw_progress()

            self.viewer.render(return_rgb_array=mode == 'rgb_array')
            import time
            time.sleep(1)

        return

    def draw_grid(self):
        grid = []
        linewidth = 2
        for x in range(self.state.shape[0] + 1):
            grid.append(
                rendering.make_polyline([
                    (x * self.draw_scale + self.draw_start_x, self.draw_start_y),
                    (x * self.draw_scale + self.draw_start_x,
                     self.draw_start_y + self.state.shape[1] * self.draw_scale)])
            )
            grid[-1].set_linewidth(linewidth)
            self.viewer.add_geom(grid[-1])

        for y in range(self.state.shape[1] + 1):
            grid.append(
                rendering.make_polyline([
                    (self.draw_start_x, y * self.draw_scale + self.draw_start_y),
                    (self.draw_start_x + self.state.shape[0] * self.draw_scale,
                     y * self.draw_scale + self.draw_start_y)])
            )
            grid[-1].set_linewidth(linewidth)
            self.viewer.add_geom(grid[-1])

        return grid

    def draw_obstacles(self):
        drawn_obstacles = []
        for obstacle in self.components['obstacles']:
            drawn_obstacles.append(rendering.FilledPolygon(
                [(0, 0), (0, self.draw_scale), (self.draw_scale, self.draw_scale), (self.draw_scale, 0)]))
            x, y = obstacle[0], obstacle[1]
            drawn_obstacles[-1].add_attr(
                rendering.Transform(
                    translation=(self.draw_start_x + x * self.draw_scale, y * self.draw_scale + self.draw_start_y))
            )
            self.viewer.add_geom(drawn_obstacles[-1])
        return drawn_obstacles

    def draw_agents(self, type_='draw', fname=None):
        drawn_agent = []
        for agent in self.components['agents']:
            drawn_agent.append(rendering.Transform())

            if type_ == 'draw':
                # arms
                arms = rendering.FilledPolygon(
                    [(-0.45 * self.draw_scale, -0.1 * self.draw_scale),
                     (0.45 * self.draw_scale, -0.1 * self.draw_scale), \
                     (0.45 * self.draw_scale, 0.1 * self.draw_scale), (-0.45 * self.draw_scale, 0.1 * self.draw_scale)])
                arms.add_attr(drawn_agent[-1])
                self.viewer.add_geom(arms)

                # body border
                body_border = rendering.make_circle(0.35 * self.draw_scale)
                body_border.add_attr(drawn_agent[-1])
                self.viewer.add_geom(body_border)

                # colored body
                body = rendering.make_circle(0.3 * self.draw_scale)
                if agent.type in self.agents_color.keys():
                    color_name = self.agents_color[agent.type]
                    body.set_color(self.colors[color_name][0], self.colors[color_name][1], self.colors[color_name][2])
                else:
                    body.set_color(self.colors['lightgrey'][0], self.colors['lightgrey'][1],
                                   self.colors['lightgrey'][2])
                body.add_attr(drawn_agent[-1])
                self.viewer.add_geom(body)

                # eyes border
                left_eye_border = rendering.make_circle(0.15 * self.draw_scale)
                left_eye_border.add_attr(
                    rendering.Transform(translation=(-0.125 * self.draw_scale, 0.25 * self.draw_scale))
                )
                left_eye_border.add_attr(drawn_agent[-1])
                self.viewer.add_geom(left_eye_border)

                right_eye_border = rendering.make_circle(0.15 * self.draw_scale)
                right_eye_border.add_attr(
                    rendering.Transform(translation=(0.125 * self.draw_scale, 0.25 * self.draw_scale))
                )
                right_eye_border.add_attr(drawn_agent[-1])
                self.viewer.add_geom(right_eye_border)

                # eyes
                left_eye = rendering.make_circle(0.1 * self.draw_scale)
                left_eye.add_attr(
                    rendering.Transform(translation=(-0.125 * self.draw_scale, 0.25 * self.draw_scale))
                )
                left_eye.set_color(self.colors['white'][0], self.colors['white'][1], self.colors['white'][2])
                left_eye.add_attr(drawn_agent[-1])
                self.viewer.add_geom(left_eye)

                right_eye = rendering.make_circle(0.1 * self.draw_scale)
                right_eye.add_attr(
                    rendering.Transform(translation=(0.125 * self.draw_scale, 0.25 * self.draw_scale))
                )
                right_eye.set_color(self.colors['white'][0], self.colors['white'][1], self.colors['white'][2])
                right_eye.add_attr(drawn_agent[-1])
                self.viewer.add_geom(right_eye)

                # retina
                left_retina = rendering.make_circle(0.05 * self.draw_scale)
                left_retina.add_attr(
                    rendering.Transform(translation=(-0.125 * self.draw_scale, 0.3 * self.draw_scale))
                )
                left_retina.add_attr(drawn_agent[-1])
                self.viewer.add_geom(left_retina)

                right_retina = rendering.make_circle(0.05 * self.draw_scale)
                right_retina.add_attr(
                    rendering.Transform(translation=(0.125 * self.draw_scale, 0.3 * self.draw_scale))
                )
                right_retina.add_attr(drawn_agent[-1])
                self.viewer.add_geom(right_retina)

                # name
                x_shift = -0.1 if agent.direction == 'E' else 0
                x_shift += 0.1 if agent.direction == 'W' else 0
                y_shift = -0.1 if agent.direction == 'N' else 0
                y_shift += 0.1 if agent.direction == 'S' else 0

                label = DrawText(pyglet.text.Label(str(agent.index), font_size=int(0.25 * self.draw_scale),
                                                   x=x_shift * self.draw_scale, y=y_shift * self.draw_scale,
                                                   anchor_x='center', anchor_y='center', color=(0, 0, 0, 255)))
                label.add_attr(drawn_agent[-1])
                self.viewer.add_geom(label)

            elif type_ == 'figure':
                try:
                    with open(fname):
                        figure = rendering.Image(fname, \
                                                 width=0.9 * self.draw_scale, height=0.9 * self.draw_scale)
                        if agent.type in self.agents_color.keys():
                            color_name = self.agents_color[agent.type]
                            figure.set_color(self.colors[color_name][0], self.colors[color_name][1],
                                             self.colors[color_name][2])
                        else:
                            figure.set_color(self.colors['lightgrey'][0], self.colors['lightgrey'][1],
                                             self.colors['lightgrey'][2])
                        figure.add_attr(drawn_agent[-1])
                        self.viewer.add_geom(figure)

                        # name
                        x_shift = 0.4 if agent.direction == 'E' else 0.5
                        x_shift += 0.1 if agent.direction == 'W' else 0
                        y_shift = 0.4 if agent.direction == 'N' else 0.5
                        y_shift += 0.1 if agent.direction == 'S' else 0

                        label = DrawText(pyglet.text.Label(str(agent.index), font_size=int(0.25 * self.draw_scale),
                                                           x=x_shift * self.draw_scale, y=y_shift * self.draw_scale,
                                                           anchor_x='center', anchor_y='center', color=(0, 0, 0, 255)))
                        label.add_attr(drawn_agent[-1])
                        self.viewer.add_geom(label)

                except FileNotFoundError as e:
                    raise e
            else:
                raise NotImplementedError

        return drawn_agent

    def draw_tasks(self, type_='box', fname=None):
        drawn_tasks, shift = [], 0
        for task in self.components['tasks']:
            if not task.completed:
                drawn_tasks.append(rendering.Transform())
                if type_ == 'box':
                    shift = 0.1
                    box = rendering.PolyLine([
                        (0.1 * self.draw_scale, 0.4 * self.draw_scale),
                        (0.5 * self.draw_scale, 0.4 * self.draw_scale),
                        (0.5 * self.draw_scale, 0.1 * self.draw_scale),
                        (0.1 * self.draw_scale, 0.1 * self.draw_scale),
                        (0.1 * self.draw_scale, 0.4 * self.draw_scale),
                        (0.35 * self.draw_scale, 0.65 * self.draw_scale),
                        (0.75 * self.draw_scale, 0.65 * self.draw_scale),
                        (0.75 * self.draw_scale, 0.35 * self.draw_scale),
                        (0.5 * self.draw_scale, 0.1 * self.draw_scale),
                        (0.5 * self.draw_scale, 0.4 * self.draw_scale),
                        (0.75 * self.draw_scale, 0.65 * self.draw_scale)
                    ], close=False)
                    box.set_linewidth(2)
                    box.add_attr(drawn_tasks[-1])
                    self.viewer.add_geom(box)
                elif type_ == 'figure':
                    shift = 0.5
                    try:
                        with open(fname):
                            figure = rendering.Image(fname, \
                                                     width=0.9 * self.draw_scale, height=0.9 * self.draw_scale)
                            figure.add_attr(drawn_tasks[-1])
                            self.viewer.add_geom(figure)
                    except FileNotFoundError as e:
                        raise e
                else:
                    raise NotImplementedError
            else:
                drawn_tasks.append(None)

        return drawn_tasks, shift

    def draw_fog(self):
        fog = FilledPolygonCv4([(0, 0), (0, self.state.shape[1] * self.draw_scale),
                                (self.state.shape[0] * self.draw_scale, self.state.shape[1] * self.draw_scale),
                                (self.state.shape[0] * self.draw_scale, 0)])
        fog.set_color(self.colors['lightgrey'][0], self.colors['lightgrey'][1], self.colors['lightgrey'][2], 0.6)
        fog.add_attr(rendering.Transform(translation=(self.draw_start_x, self.draw_start_y)))
        self.viewer.geoms.insert(0, fog)

    def draw_vision(self, agent, transagent):
        vision = make_circleCv4(radius=self.draw_scale *
                                       np.sqrt(self.state.shape[0] ** 2 + self.state.shape[1] ** 2) * agent.radius,
                                angle=(2 * np.pi) * agent.angle)
        vision.set_color(self.colors['white'][0], self.colors['white'][1], self.colors['white'][2], 0.7)
        vision.add_attr(transagent)
        self.viewer.geoms.insert(1, vision)

    def draw_progress(self):
        if self.episode != 0:
            self.viewer.geoms.pop()

        progress = 100 * ([t.completed for t in self.components['tasks']].count(True) / len(self.components['tasks']))
        progress_label = DrawText(pyglet.text.Label(
            'Episode: ' + str(self.episode) + ' | Progress: ' + str(progress) + '%', \
            font_size=int(0.25 * self.draw_scale),
            x=self.draw_start_x, y=(1 + self.state.shape[1]) * self.draw_scale,
            anchor_x='left', anchor_y='center', color=(0, 0, 0, 255)))
        self.viewer.add_geom(progress_label)


def make_circleCv4(radius=10, angle=2 * np.pi, res=30):
    points = [(0, 0)]
    for i in range(res + 1):
        ang = (np.pi - angle) / 2 + (angle * (i / res))
        points.append((np.cos(ang) * radius, np.sin(ang) * radius))
    return FilledPolygonCv4(points)
