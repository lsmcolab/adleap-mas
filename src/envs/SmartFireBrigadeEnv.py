from importlib import import_module
from gym import spaces

from datetime import datetime
import numpy as np
import random as rd
import sys

sys.path.append('../src/reasoning')
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Ad-hoc 
"""
class Agent(AdhocAgent):
    """Agent : Component of the Environment. Derives from AdhocAgent Class
    """
    def __init__(self, index, atype, position, direction, radius, angle, comm_protocol='broadcast'):
        super(Agent, self).__init__(index, atype)

        # agent movement parameters
        self.position = position
        self.direction = direction

        self.velocity = 0.0
        self.velocity_threshold = 10.0
        self.acc = 0.1

        # agent vision parameters
        self.radius = radius
        self.angle = angle

        # agent fire brigade parameters
        self.brigade = []
        self.resources = {'water':100,'battery':100}
        self.extinguish_range = 20

        # defining communication parameters
        from importlib import import_module
        try:
            module = import_module('src.communication.'+comm_protocol)
        except:
            raise NotImplemented
        communication_protocol = getattr(module, 'get_'+comm_protocol)
        self.communication = communication_protocol()
        self.type = atype
        # agent memory
        self.memory = {'fire':[],'agent':[]}

    def copy(self):
        
        # 1. Initialising the agent
        copy_agent = Agent(self.index, self.type, self.position, \
                           self.direction, self.radius, self.angle)

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters
        copy_agent.memory = self.memory
        
        return copy_agent


class Fire(object):
    """Fire : Task of the environemnt.
    """
    def __init__(self, position, level, time_constraint=True):
        # task parameters
        self.position = position
        self.spreading_level = float(level)

        # task simulation parameters
        self.extinguished = False
        self.trying = []

        # time dependent functions
        # - Parker2014aamas: Tasks with Cost Growing over Time and Agent Reallocation Delays
        # ft: current task cost
        # ht: current task growing cost
        # wt: current work impact
        self.time_constraint = time_constraint

        self.ft = lambda ft,ht,wt: ft + (ht - wt)
        if self.time_constraint:
            self.ht = lambda ft: ft*0.001
        else:
            self.ht = lambda ft: ft*0.0

    def copy(self):
        # 1. Initialising the copy task
        copy_task = Fire(self.position, self.spreading_level, self.time_constraint)

        # 2. Copying the parameters
        copy_task.extinguished = self.extinguished
        copy_task.trying = [a for a in self.trying]

        return copy_task
    
    def update(self,wt):
        ft = self.spreading_level
        ht = self.ht(self.spreading_level)
        self.spreading_level = self.ft(ft,ht,wt)
        
        if self.spreading_level < 0.2:
            self.extinguished = True

        if self.extinguished:
            self.spreading_level = 0
"""
    Customising the SmartFireBrigade Env
"""
"""
1. VISIBILITY ADN GET METHODS
"""
# This method returns the visible tasks positions
def get_visible_agents_and_tasks(state, agent, components):
    # 1. Defining the agent vision parameters
    direction = agent.direction
    radius = agent.radius
    angle = agent.angle

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

"""
2. ENVIRONMENT UPDATE AND VERIFICATION METHODS
"""
def extinguish_fire(env, trying_extinguish):
    extinguish_reward = 0

    # 1. Calculating the work impact
    wt = {}
    for agent in trying_extinguish:
        for fire in env.components['fire']:
            distance = np.linalg.norm(np.array(agent.position) - \
                                        np.array(fire.position))
                                    
            if distance <= agent.extinguish_range and\
             agent.resources['water'] > 0 and agent.resources['battery'] > 0:
                if fire.position not in wt:
                    wt[fire.position] = 0.05
                else:
                    wt[fire.position] += 0.05

                extinguish_reward += 1

    for fire in env.components['fire']:
        if fire.position in wt:
            fire.update(wt[fire.position])
        else:
            fire.update(0)

    # 2. Updating the fire spreading level
    return extinguish_reward

def establish_communication(trying_communication):
    # 1. Defining the communication control
    from src.communication.control import ProtocolHandler
    protocolHandler = ProtocolHandler()
    # 2. First, we will check the team formation
    connection_pairs = protocolHandler.check_connection(trying_communication)

    # 3. and then, the message exchange
    # a. updating the brigade formation (hand-shaking formation)
    brigades = []
    for i in range(len(trying_communication)):
        # - checking if both agents agreed in forming a team
        connection_target = connection_pairs[i]
        if connection_target!=-1 and connection_pairs[connection_target] == i and  connection_target != i and\
        trying_communication[connection_target].index not in trying_communication[i].brigade:
            trying_communication[i].brigade.append(trying_communication[connection_target].index)

        # - saving the current brigade in the vector
        brigades.append([index for index in trying_communication[i].brigade])

    # b. handling the communication    
    protocolHandler.check_communication(trying_communication, brigades)
        
    return 0

def do_action(actions, env):
    next_state, info = None, {}
    positions, directions, costs = {}, {}, {}

    trying_extinguish, trying_communication = [], []
    
    ###
    # MOVEMENT
    ###
    for agent in env.components['agents']:
        # 1. Position and direction
        # - defining the agents new position and direction
        # checking agent status
        pos, dir = agent.position, agent.direction
        costs[agent.index] = {'battery':0.0,'water':0.0}

        if isinstance(actions,dict):
            agent_action = actions[agent.index]
        else:
            agent_action = actions

        # VELOCITY
        # -> NOOP
        if agent_action == 0 or\
        agent_action == None: 
            positions[agent.index] = pos
            directions[agent.index] = dir
            agent.velocity = agent.velocity*0.95

        # -> UP (ACCELERATE)
        elif agent_action == 1:
            if agent.velocity >= 1:
                agent.velocity = agent.velocity*(1 + agent.acc) \
                    if agent.velocity*(1 + agent.acc) <= agent.velocity_threshold \
                        else agent.velocity_threshold
            else:
                agent.velocity = 1.0
            costs[agent.index]['battery'] += env.move_cost*1.1
                    
        # -> DOWN (BREAK)
        elif agent_action == 3 :
            agent.velocity = agent.velocity*(1 - agent.acc)
            costs[agent.index]['battery'] += env.move_cost

        positions[agent.index] = (
            pos[0] + env.move_scale*agent.velocity*np.cos(dir),
            pos[1] + env.move_scale*agent.velocity*np.sin(dir))

        # ROTATION
        # -> TURN RIGHT
        if agent_action == 2:
            directions[agent.index] = (dir - np.pi*env.rotation_scale) % (2*np.pi)
            costs[agent.index]['battery'] += env.move_cost

        # -> TURN LEFT
        elif agent_action == 4 :
            directions[agent.index] = (dir + np.pi*env.rotation_scale) % (2*np.pi)  
            costs[agent.index]['battery'] += env.move_cost 

        else:
            directions[agent.index] = dir

        if agent_action == 5: # Extinguish
            trying_extinguish.append(agent)
            costs[agent.index]['battery'] += env.extinguish_battery
            costs[agent.index]['water'] += env.extinguish_water

        if agent_action == 6: # Communicate
            trying_communication.append(agent)
            costs[agent.index]['battery'] += env.communicate_battery

        # - checking if every agent is still reachable
        out_brigade  = []
        for index in agent.brigade:
            for other_agent in env.components['agents']:
                if other_agent.index == index:
                    if distance(agent.position, other_agent.position) > \
                    agent.communication.max_distance:
                        out_brigade.append(index)

        for index in out_brigade:
            agent.brigade.remove(index)

    # - trying extinguish
    info['extinguish_reward'] = extinguish_fire(env, trying_extinguish)

    # - trying communication
    info['communication_reward'] = establish_communication(trying_communication)

    # 3. Updating state
    for agent in env.components['agents']:
        if agent.resources['battery'] > 0:
            if 0 < positions[agent.index][0] < env.dim[0]\
            and 0 < positions[agent.index][1] < env.dim[1]:
                agent.position = positions[agent.index]
                agent.direction = directions[agent.index]
                agent.resources['battery'] -= costs[agent.index]['battery']
                agent.resources['water'] -= costs[agent.index]['water']

                cost_coef = 0.2
                info['costs_reward'] = 0#-cost_coef*(costs[agent.index]['battery'] + costs[agent.index]['water'])
            else:
                agent.velocity = 0
                
                new_position = [1,1]
                if positions[agent.index][0] <= 0:
                    new_position[0] = 1
                elif positions[agent.index][0] >= env.dim[0]:
                    new_position[0] = env.dim[0] - 1

                if positions[agent.index][1] <= 0:
                    new_position[1] = 1
                elif positions[agent.index][1] >= env.dim[1]:
                    new_position[1] = env.dim[1] - 1

                agent.direction = directions[agent.index]
                agent.resources['battery'] -= costs[agent.index]['battery']
                agent.resources['water'] -= costs[agent.index]['water']
        
        if agent.resources['battery'] < 0:
            agent.resources['battery'] = 0
        if agent.resources['water'] < 0:
            agent.resources['water']
    
    env.update()
    next_state = env.copy()

    return next_state, info

def spawn_fire(real_env, max_simultaneous_fire=10):
    if len(real_env.components['fire']) <= max_simultaneous_fire:
        spawn_allowed = ((datetime.now() - real_env.last_spawn).total_seconds() > real_env.spawn_delay)
        coin = rd.uniform(0,1)
        
        if spawn_allowed and coin < real_env.spawn_probability :
            real_env.last_spawn = datetime.now()
            real_env.components['fire'].append(Fire(
                position=(rd.randint(real_env.pad,real_env.dim[0] - real_env.pad),
                    rd.randint(real_env.pad,real_env.dim[1] - real_env.pad)),
                level = rd.randint(1, 5) ))

"""
3. ENVIRONMENT MODEL METHODS
"""
def end_condition(state):
    return False

def smartfirebrigade_transition(actions, real_env):
    # Environment step
    next_state, info = do_action(actions, real_env)

    # Fire spawn
    # - If spawn_delay is None, fire will not spawn in the environment
    # hence, the initial set of fires is the set of tasks to be accomplished
    if real_env.spawn_delay is not None:
        spawn_fire(real_env, max_simultaneous_fire=10)


    # Updating agents memory
    # TODO
    for agent in real_env.components['agents']:
        for fire in real_env.components['fire']:
            if(fire.position not in agent.memory['fire'] and is_visible(fire.position,agent.position,agent.direction,agent.radius,agent.angle)):
                agent.memory['fire'].append(fire.position)

    for agent in real_env.components['agents']:
        for agent2 in real_env.components['agents']:
            if(agent2.index not in agent.memory['agent'] and is_visible(agent2.position,agent.position,agent.direction,agent.radius,agent.angle)):
                agent.memory['agent'].append(agent2.index)

    # retuning the results
    return next_state, info


# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return 0

# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    copied_env.state = {}
    agent = copied_env.get_adhoc_agent()
    if not agent:
        return copied_env

    ###
    # VISIBILITY CHECKING
    ###
    if agent.radius is not None:
        radius = agent.radius
    else:
        radius = np.sqrt(copied_env.square_meters) * rd.uniform(0, 1)

    if agent.radius is not None:
        angle = agent.angle
    else:
        angle = 2 * np.pi * rd.uniform(0, 1)

    ###
    # BUILDING OBSERVABLE STATE
    ###
    # 1. Removing the invisible agents and tasks from environment
    invisible_agents = []
    for i in range(len(copied_env.components['agents'])):
        if not is_visible(copied_env.components['agents'][i].position,
        agent.position, agent.direction, radius, angle) and \
        copied_env.components['agents'][i] != agent:
            invisible_agents.append(i)

    invisible_tasks = []
    for i in range(len(copied_env.components['fire'])):
        if not is_visible(copied_env.components['fire'][i].position,
        agent.position, agent.direction, radius, angle) or \
        copied_env.components['fire'][i].extinguished:
            invisible_tasks.append(i)

    # 2. Building the observable environment
    copied_env.state = {'agents':[],'fire':[]}

    # a. setting the visible components
    # - agents
    for i in range(len(copied_env.components['agents'])-1, -1,-1):
        if i not in invisible_agents and copied_env.components['agents'][i].index != agent.index:
            pos = copied_env.components['agents'][i].position
            copied_env.state['agents'].append(pos)
        elif copied_env.components['agents'][i].index != agent.index:
            del copied_env.components['agents'][i]

    # - tasks
    for i in range(len(copied_env.components['fire'])-1, -1,-1):
        if i not in invisible_tasks:
            pos = copied_env.components['fire'][i].position
            copied_env.state['fire'].append(pos)
        else:
            del copied_env.components['fire'][i]

    return copied_env


"""
    SmartFireBrigade Environment
"""
class FireBrigadeState(spaces.Space, object):
    
    def __init__(self,square_meters):
        # Initialising the state space
        super(FireBrigadeState, self).__init__(
            shape=(int(np.sqrt(square_meters)),int(np.sqrt(square_meters))),dtype=np.float64)

    def sample(self, seed=None):
        state = {'agents':[],'fire':[]}
        return state

class SmartFireBrigadeEnv(AdhocReasoningEnv):
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

    agents_color = {
    }

    action_dict = {
        0:'NOOP',
        1:'UP',
        2:'RIGHT',
        3:'DOWN',
        4:'LEFT',
        5:'EXTINGUISH',
        6:'COMMUNICATE'
    }

    def __init__(self, components, square_meters=None, dim=None, spawn_delay=30, display=False):
        self.viewer = None
        self.display = display
        self.start_time = datetime.now()
        self.last_spawn = datetime.now()
        self.spawn_delay = spawn_delay
        self.spawn_probability = 0.05

        # costs
        self.move_cost = 0.01
        self.extinguish_battery = 0.02
        self.extinguish_water = 0.2
        self.communicate_battery = 0.2

        # render scale
        self.move_scale = 0.03
        self.rotation_scale = 0.005

        # Defining the Ad-hoc Reasoning Env parameters
        if square_meters is not None:
            self.square_meters = square_meters
            self.dim = (np.sqrt(square_meters),np.sqrt(square_meters))
        else:
            self.dim = dim
            self.square_meters = (dim[0]*dim[1])

        state_set = StateSet(FireBrigadeState(self.square_meters), end_condition)
        transition_function = smartfirebrigade_transition
        action_space = spaces.Discrete(7)
        reward_function = reward
        observation_space = environment_transformation

        # Initialising the Adhoc Reasoning Env
        super(SmartFireBrigadeEnv, self).__init__(state_set, \
                                               transition_function, action_space, reward_function, \
                                               observation_space, components)

        # Setting the initial components
        self.state_set.initial_components = self.copy_components(components)

        # Setting the inital state
        self.state_set.initial_state = {'agents':[],'fire':[]}
        for agent in self.state_set.initial_components['agents']:
            self.state_set.initial_state['agents'].append(agent.position)

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.smartfirebrigade.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = SmartFireBrigadeEnv(components, square_meters=self.square_meters)
        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)

        return copied_env
    
    def update(self):
        self.state = {'agents':[],'fire':[]}
        for agent in self.components['agents']:
            self.state['agents'].append(agent.position)
        for fire in self.components['fire']:
            self.state['fire'].append(fire.position)
    
    def get_logdata(self):
        data = {}
        #header = ['Time Step (s)','#Fire','Spreading Level','#Agents','Battery Level','Water Level', 'Exploration Level']
        data['time'] = (datetime.now() - self.start_time).total_seconds()
        data['nfire'] = sum([0 if fire.extinguished else 1 for fire in self.components['fire']])
        data['spreading_level'] = [fire.spreading_level for fire in self.components['fire']]
        data['nagents'] = len(self.components['agents'])
        data['battery'] = str([agent.resources['battery'] for agent in self.components['agents']])
        data['water'] = str([agent.resources['water'] for agent in self.components['agents']])
        data['ex_level'] = str(0)
        return data

    def get_actions_list(self):
        return [0,1,2,3,4,5,6]

    def get_action(self,name):
        for key in self.action_dict:
            if self.action_dict[key] == name:
                return key
        return None

    def get_adhoc_agent(self):
        for agent in self.components['agents']:
            if agent.index == self.components['adhoc_agent_index']:
                return agent
        return None
    
    def get_fire(self, position):
        for fire in self.components['fire']:
            if position[0] == fire.position[0] and\
                position[1] == fire.position[1]: 
                return fire
        return None

    def state_is_equal(self, state):
        for x in range(self.state.shape[0]):
            for y in range(self.state.shape[1]):
                if self.state[x, y] != state[x, y]:
                    return False
        return True

    def observation_is_equal(self, obs):
        observable_env = self.observation_space(self.copy())
        adhoc_agent = self.get_adhoc_agent()

        # agents
        for obs_agent in obs.state['agents']:
            print(obs_agent,observable_env.state['agents'])
            if obs_agent not in observable_env.state['agents'] and\
                obs_agent != adhoc_agent.position:
                return False

        # fire
        for obs_fire in obs.state['fire']:
            if obs_fire not in observable_env.state['fire']:
                return False

        return True

    def sample_state(self, agent, sample_p=0.1, n_sample=10):
        # 1. Defining the base simulation
        u_env = self.copy()
        obs = u_env.get_observation()

        # - setting environment components
        count = 0
        while count < n_sample:
            # - setting teammates
            if rd.uniform(0, 1) < sample_p:
                obs.state['agents'].append(\
                    ( rd.randint(0, u_env.dim[0]),\
                      rd.randint(0, u_env.dim[1]) )
                 )

            # - setting tasks
            if rd.uniform(0, 1) < sample_p:
                obs.state['fire'].append(\
                    ( rd.randint(0, u_env.dim[0]),\
                      rd.randint(0, u_env.dim[1]) )
                 )
            count += 1

        return u_env

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
        module = import_module(new_type)
        planning_method = getattr(module,  new_type + '_planning')

        _, target = \
            planning_method(obsavable_env, adhoc_agent)

        # retuning the results
        for task in self.components['tasks']:
            if task.position == target:
                return task
        return None

    def render(self, mode='human'):
        # checking if the user wants display
        if not self.display:
            raise IOError

        # importing the necessary packages
        try:
            global rendering
            from gym.envs.classic_control import rendering
            from gym.envs.classic_control.rendering import PolyLine
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
        
        ##########
        #####
        ##
        ## RENDERING THE WORLD
        ##
        #####
        ##########
        # Render the environment to the screen
        if self.viewer is None:
            self.screen_width, self.screen_height, self.pad = 1000, 1000,10
            self.scale = int(self.screen_width/np.sqrt(self.square_meters))
            self.viewer = rendering.Viewer(self.screen_width + 2*self.pad, self.screen_height + 2*self.pad)

            self.d_agents, self.d_fire = [], []
            self.timer = None

            # Drawing the environment
            # (A) Scenario/map
            background = self.draw_map()
            self.viewer.add_geom(background)

            # (B) Firefighters Drones
            agents = self.draw_agents()
            for agent in agents:
                self.d_agents.append(agent)   
            self.batteries, self.waters = [], []

            # (C) Brigade formation
            self.brigade_formation = [] 
            for agent in agents:
                self.brigade_formation.append([])
        
        # checking the agents
        for i in range(len(self.components['agents'])):
            # bettery and water level
            b_percentage = self.components['agents'][i].resources['battery']/100
            
            battery = FilledPolygonCv4([\
                (0*self.scale,0*self.scale),\
                (0*self.scale,1*self.scale),\
                (2*b_percentage*self.scale,1*self.scale),\
                (2*b_percentage*self.scale,0*self.scale)])
            battery.add_attr(rendering.Transform(translation=(-0.75*self.scale, self.scale)))
            battery.set_color(0, 1, 0, 0.8)
            battery.add_attr(self.d_agents[i])

            if len(self.batteries) > i:
                if self.batteries[i] in self.viewer.geoms:
                    self.viewer.geoms.remove(self.batteries[i])
                self.viewer.add_geom(battery) 
                self.batteries[i] = battery 
            else:
                self.viewer.add_geom(battery) 
                self.batteries.append(battery) 

            water = FilledPolygonCv4([\
                (0*self.scale,0*self.scale),\
                (0*self.scale,1*self.scale),\
                (2*self.components['agents'][i].resources['water']/100*self.scale,1*self.scale),\
                (2*self.components['agents'][i].resources['water']/100*self.scale,0*self.scale)])
            water.add_attr(rendering.Transform(translation=(-0.75*self.scale, 2*self.scale)))
            water.set_color(0, 0, 1, 0.8)
            water.add_attr(self.d_agents[i]) 

            if len(self.waters) > i:
                if self.waters[i] in self.viewer.geoms:
                    self.viewer.geoms.remove(self.waters[i])
                self.viewer.add_geom(water) 
                self.waters[i] = water 
            else:
                self.viewer.add_geom(water) 
                self.waters.append(water) 

            # agents
            self.d_agents[i].set_translation(0,0)
            self.d_agents[i].set_rotation(self.components['agents'][i].direction)
            self.d_agents[i].set_translation(
                self.scale*self.components['agents'][i].position[0] + self.pad,
                self.scale*self.components['agents'][i].position[1] + self.pad) 
            
        # checking the fire
        for i in range(len(self.components['fire'])):
            if i >= len(self.d_fire):
                fire = self.draw_fire(self.components['fire'][i].position)
                self.viewer.add_geom(fire)
                self.d_fire.append(fire)

            elif self.components['fire'][i].extinguished:
                self.d_fire[i].attrs[-1].set_scale(0.0, 0.0)

            else:
                position = self.components['fire'][i].position 
                level = self.components['fire'][i].spreading_level
                self.d_fire[i].attrs[-1].set_translation(0,0)
                self.d_fire[i].attrs[-1].set_scale(level, level)
                self.d_fire[i].attrs[-1].set_translation(\
                    self.scale*position[0] + self.pad,
                    self.scale*position[1] + self.pad)

        # checking brigades
        for i in range(len(self.components['agents'])):
            points = []

            # all connections to agent i
            for agent in self.components['agents']:
                if agent.index in self.components['agents'][i].brigade:
                    points.append((self.scale*agent.position[0] + self.pad,
                                    self.scale*agent.position[1] + self.pad))
                    points.append((self.scale*self.components['agents'][i].position[0] + self.pad,
                        self.scale*self.components['agents'][i].position[1] + self.pad))

            # removing old formation
            if self.brigade_formation[i] in self.viewer.geoms:
                self.viewer.geoms.remove(self.brigade_formation[i])

            # drawing 
            self.brigade_formation[i] = None
            if len(points) > 1:
                self.brigade_formation[i] = PolyLine(points,close=False)
                self.brigade_formation[i].set_linewidth(3)
                self.viewer.add_geom(self.brigade_formation[i])
    
        # timer
        self.timer = self.draw_timer()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def draw_map(self,mode='img'):
        if mode == 'draw':
            background = rendering.FilledPolygon([
                (0 + self.pad, 0 + self.pad), 
                (0 + self.pad, self.screen_height + self.pad ), 
                (self.screen_width + self.pad, self.screen_height + self.pad), 
                (self.screen_width + self.pad, 0 + self.pad)])
            background.set_color(0.2,0.7,0.2)
        else:
            try:
                with open('imgs/smartfirebrigade/random_forest.jpg'):
                    background = rendering.Image('imgs/smartfirebrigade/random_forest.jpg', \
                                width=self.screen_width, height=self.screen_height)
                    background.add_attr(rendering.Transform(
                        translation=(int(self.screen_width)/2 + self.pad,
                                    int(self.screen_height)/2 + self.pad)))
            except FileNotFoundError as e:
                raise e
        return background

    def draw_agents(self):
        agents = []
        for agent in self.components['agents']:
            agents.append(rendering.Transform())

            viscom_range = make_circleCv4(radius=self.scale*agent.radius,
                                             angle=agent.angle,filled=True)
            viscom_range.set_color(1, 0, 0, 0.2)
            viscom_range.add_attr(rendering.Transform(rotation=np.pi))
            viscom_range.add_attr(agents[-1])
            self.viewer.add_geom(viscom_range)

            drone = rendering.FilledPolygon([
                (-0.5*self.scale,-0.5*self.scale),
                ( 0.5*self.scale,-0.5*self.scale),
                ( 0.5*self.scale, 0.5*self.scale),
                (-0.5*self.scale, 0.5*self.scale)])
            drone.add_attr(agents[-1])
            self.viewer.add_geom(drone)

            propeller1 = make_circleCv4(0.35*self.scale)
            propeller1.add_attr(rendering.Transform(translation=(-0.45*self.scale,-0.45*self.scale)))
            propeller1.set_color(0, 0, 0, 0.6)
            propeller1.add_attr(agents[-1])
            self.viewer.add_geom(propeller1)

            propeller2 = make_circleCv4(0.35*self.scale)
            propeller2.add_attr(rendering.Transform(translation=( 0.45*self.scale,-0.45*self.scale)))
            propeller2.set_color(0, 0, 0, 0.6)
            propeller2.add_attr(agents[-1])
            self.viewer.add_geom(propeller2)

            propeller3 = make_circleCv4(0.35*self.scale)
            propeller3.add_attr(rendering.Transform(translation=( 0.45*self.scale, 0.45*self.scale)))
            propeller3.set_color(0, 0, 0, 0.6)
            propeller3.add_attr(agents[-1])
            self.viewer.add_geom(propeller3)

            propeller4 = make_circleCv4(0.35*self.scale)
            propeller4.add_attr(rendering.Transform(translation=(-0.45*self.scale, 0.45*self.scale)))
            propeller4.set_color(0, 0, 0, 0.6)
            propeller4.add_attr(agents[-1])
            self.viewer.add_geom(propeller4)       
                             
        return agents
    
    def draw_fire(self,position):
        try:
            with open('imgs/smartfirebrigade/fire.png'):
                fire = rendering.Image('imgs/smartfirebrigade/fire.png', \
                                            width=25, height=25)
        except FileNotFoundError as e:
            raise e
            
        fire.add_attr(rendering.Transform(
            translation=(int(self.scale*position[0]),
                         int(self.scale*position[1])))) 
        return fire

    def draw_timer(self):
        if self.timer in self.viewer.geoms:
            self.viewer.geoms.remove(self.timer)

        from datetime import datetime
        execution_time = (datetime.now() - self.start_time).total_seconds()
        timer = DrawText(pyglet.text.Label(
            '%.3f s' %(execution_time),\
            font_size=18, bold=True, 
            x=0.8*self.screen_width, y= 25,
            anchor_x='right', anchor_y='center', 
            color=(100, 0, 0, 255)))
        self.viewer.add_geom(timer)
        return timer

def make_circleCv4(radius=10, angle=2*np.pi, res=30, filled=True):
    points = [(0, 0)]
    for i in range(res + 1):
        ang = (np.pi - (angle) / 2) + (angle * (i / res))
        points.append((np.cos(ang) * radius, np.sin(ang) * radius))
    if filled:
        return FilledPolygonCv4(points)
    else:
        return rendering.PolyLine(points,True)
