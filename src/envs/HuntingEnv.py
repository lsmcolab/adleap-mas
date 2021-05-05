from datetime import datetime
from copy import deepcopy
import gym
from gym import error, spaces
from gym.envs.classic_control import rendering
import numpy as np
import random as rd

from .AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Rendering 
"""
try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

try:
    from pyglet.gl import glBegin, glEnd, GL_QUADS, GL_POLYGON, GL_TRIANGLES, glVertex3f
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')


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


def make_circleCv4(radius=10, angle= 2*np.pi, res=30):
    points = [(0, 0)]
    for i in range(res+1):
        ang = (np.pi - angle)/2 + (angle * (i / res))
        points.append((np.cos(ang) * radius, np.sin(ang) * radius))
    return FilledPolygonCv4(points)


class DrawText(rendering.Geom):
    def __init__(self, label: pyglet.text.Label):
        rendering.Geom.__init__(self)
        self.label = label

    def render1(self):
        self.label.draw()

"""
    Ad-hoc 
"""
class Agent(AdhocAgent):
    """Agent : Main reasoning Component of the Environment. Derives from AdhocAgent Class
    """
    def __init__(self, index, atype, position, direction, radius, angle, level):

        super(Agent, self).__init__(index,atype)

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
        copy_agent.next_action =  self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters

        return copy_agent

    def show(self):
        print(self.index,self.type,':',self.position,self.direction,self.radius,self.angle,self.level)


class Task(AdhocAgent):
    """Agent : Main reasoning Component of the Environment. Derives from AdhocAgent Class
    """

    def __init__(self, index, atype, position, direction, radius, angle, level):
        super(Task, self).__init__(index, atype)

        # agent parameters
        self.position = position
        self.direction = direction
        self.radius = radius
        self.angle = angle
        self.level = level
        self.completed = False
        self.trying = []

    def copy(self):
        # 1. Initialising the agent
        copy_task = Task(self.index, self.type, self.position, \
                           self.direction, self.radius, self.angle, self.level)

        # 2. Copying the parameters
        copy_task.next_action = self.next_action
        copy_task.target = None if self.target is None else self.target
        copy_task.smart_parameters = self.smart_parameters
        copy_task.completed = self.completed
        copy_task.trying = [a for a in self.trying]

        return copy_task

    def show(self):
        print(self.index, self.type, ':', self.position, self.direction, self.radius, self.angle, self.level)
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
            direction = env.action_space.sample()

        if a.radius is not None:
            radius = np.sqrt(env.state.shape[0]**2 + env.state.shape[1]**2)*a.radius
        else:
            radius = np.sqrt(env.state.shape[0]**2 + env.state.shape[1]**2)*rd.uniform(0,1)

        if a.radius is not None:
            angle = 2*np.pi*a.angle
        else:
            angle = 2*np.pi*rd.uniform(0,1)

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
    radius = np.sqrt(state.shape[0]**2 + state.shape[1]**2) * agent.radius
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
    if(y==0 and x==0):
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




