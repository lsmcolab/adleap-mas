import os
from importlib import import_module
from pickle import GLOBAL
from gym import Space, spaces
from multiprocessing import Process
import numpy as np
import random as rd
import sys
from time import time

sys.path.append('../src/reasoning')
from src.envs.AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet
from src.communication.control import ProtocolHandler

FIRE_UPDATE = 1 # 1 second
MAX_SIMULTANEOUS_FIRE = 10 # 10 max fires in the env
SPAWN_DELAY = 2*60 # 2 minutes to spawn a new fire
SPAWN_PROBABILITY = 0.3 # new fires will appear with probability 0.3
FIRE_EXTINGUISH_DISTANCE = 10 # 10 meters

N_FIRE_SAMPLE = 5 # 5 fires
POLICY_MAX_STEPS = np.inf # follow a policy until the agent chooses another one
"""
    If you want, you can run experiments with limited number of steps when performing a
    policy by changing the above line:
        POLICY_MAX_STEPS = X # do X steps (max) when the agent chooses a policy to perform
"""

"""
    COMPONENTS
"""
class Agent(AdhocAgent):
    """Agent : Component of the Environment. Derives from AdhocAgent Class
    """
    def __init__(self, index, atype, radius=30, angle=2*np.pi):
        super(Agent, self).__init__(index, atype)

        # agent movement parameters
        self.position = (5,5)
        self.direction = np.pi/2

        # agent vision parameters
        self.radius = radius
        self.angle = angle

        # agent fire brigade parameters
        self.resources = {'water':100,'battery':100}
        self.communication_protocol = 'p2p'
        module = import_module('src.communication.'+self.communication_protocol)
        self.communication = getattr(module,'get_broadcast')()
        
        # agent smart parameters
        self.type = atype
        self.memory = {'fire':[],'agents':[]}

    def copy(self):
        # 1. Initialising the agent
        copy_agent = Agent(self.index, self.type, self.radius, self.angle)
        copy_agent.position = (self.position[0],self.position[1])
        copy_agent.direction = self.direction

        # 2. Copying the parameters
        copy_agent.next_action = self.next_action
        copy_agent.target = None if self.target is None else self.target
        copy_agent.smart_parameters = self.smart_parameters
        copy_agent.memory = self.copy_memory()
        copy_agent.communication_protocol = self.communication_protocol
        copy_agent.communication = self.communication

        return copy_agent

    def update_memory(self, obs_env):
        # agents
        print("Before : ",self.index,self.memory)
        for x in obs_env.components['agents']:
            if x.index != self.index:
                if x.index not in [e[0] for e in self.memory['agents']]:
                    self.memory['agents'].append((x.index,x.position[0],x.position[1]))
                else:
                    idx = [e[0] for e in self.memory['agents']].index(x.index)
                    self.memory['agents'][idx] = (x.index,x.position[0],x.position[1])
        
        missed = []
        for x in self.memory['agents']:
            if is_visible((x[1],x[2]),self.position,self.direction,self.radius,self.angle )\
             and obs_env.get_agent((x[1],x[2])) is None:
                missed.append(x)

        for i in range(len(self.memory['agents'])-1, -1,-1):
            if self.memory['agents'][i] in missed:
                del self.memory['agents'][i]  

        # fire
        for x in obs_env.components['fire']:
            if x.position not in self.memory['fire']:
                self.memory['fire'].append((x.position[0],x.position[1]))
        
        extinguishes = []
        for x in self.memory['fire']:
            if is_visible(x,self.position,self.direction,self.radius,self.angle)\
             and obs_env.get_fire(x) is None:
                extinguishes.append(x)

        for i in range(len(self.memory['fire'])-1, -1,-1):
            if self.memory['fire'][i] in extinguishes:
                del self.memory['fire'][i] 
        print("After : ",self.memory)
    
    def copy_memory(self):
        memory = {'fire':[],'agents':[]}
        for a in self.memory['agents']:
            memory['agents'].append((a[0],a[1],a[2]))
        for f in self.memory['fire']:
            memory['fire'].append((f[0],f[1]))
        return memory

class Fire(object):
    """Fire : Task of the environemnt.
    """
    def __init__(self, position, level, time_constraint=True):
        # task parameters
        self.position = position
        self.spreading_level = float(level)
        self.spreading_limit = 5

        # task simulation parameters
        self.extinguished = False
        self.expanded = False

        # time dependent functions
        # - Parker2014aamas: Tasks with Cost Growing over Time and Agent Reallocation Delays
        # ft: current task cost
        # ht: current task growing cost
        # wt: current work impact
        self.time_constraint = time_constraint

        self.ft = lambda ft,ht,wt: ft + (ht - wt)
        if self.time_constraint:
            self.ht = lambda ft: ft*0.01
        else:
            self.ht = lambda ft: ft*0.0

    def copy(self):
        # 1. Initialising the copy fire
        copy_fire = Fire(self.position, self.spreading_level, self.time_constraint)

        # 2. Copying the parameters
        copy_fire.extinguished = self.extinguished
        copy_fire.expanded = self.expanded

        return copy_fire
    
    def update(self,wt):
        extinguihed, expanded = False, False
        if wt == 0 and self.expanded:
            return extinguihed, expanded

        ft = self.spreading_level
        ht = self.ht(self.spreading_level)
        self.spreading_level = self.ft(ft,ht,wt)
        
        if self.spreading_level < 0.2:
            self.spreading_level = 0
            
            self.extinguished = True
            extinguihed = True
            return extinguihed, expanded
        
        elif self.spreading_level > self.spreading_limit:
            self.expanded = True
            expanded = True
            return extinguihed, expanded

        return extinguihed, expanded











"""
1. VISIBILITY AND GET METHODS
"""
# This method returns the visible tasks positions
def get_visible_agents_and_tasks(state, agent):
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
def do_control_action(actions, env):
    next_state, info = None, {}

    positions, directions, costs = {}, {}, {}
    trying_extinguish,trying_communicate = [],[]

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
            if agent.index not in actions:
                agent_action = 0
            else:
                agent_action = actions[agent.index]
        else:
            agent_action = actions

        # MOVEMENT
        # -> NOOP
        if agent_action == 0 or\
        agent_action == None: 
            positions[agent.index] = pos
            directions[agent.index] = dir
        # -> North
        elif agent_action == 1:
            costs[agent.index]['battery'] += env.move_cost
            positions[agent.index] = (pos[0],pos[1] + env.move_length)
            directions[agent.index] = (1/2)*np.pi
        # -> East
        elif agent_action == 2:
            costs[agent.index]['battery'] += env.move_cost
            positions[agent.index] = (pos[0]+env.move_length,pos[1])
            directions[agent.index] = (0)*np.pi
        # -> South
        elif agent_action == 3:
            costs[agent.index]['battery'] += env.move_cost
            positions[agent.index] = (pos[0],pos[1]-env.move_length)
            directions[agent.index] = (3/2)*np.pi
        # -> West
        elif agent_action == 4:
            costs[agent.index]['battery'] += env.move_cost
            positions[agent.index] = (pos[0]-env.move_length,pos[1])
            directions[agent.index] = (1)*np.pi

        # EXTINGUISH
        elif agent_action == 5: # Extinguish
            trying_extinguish.append(agent)
            positions[agent.index] = pos
            directions[agent.index] = dir
            costs[agent.index]['battery'] += env.extinguish_battery
            costs[agent.index]['water'] += env.extinguish_water
        
        else:
            costs[agent.index]['battery'] += agent.communication.cost
            trying_communicate.append(agent)
            positions[agent.index] = pos
            directions[agent.index] = dir

    # - trying extinguish
    info['extinguish_reward'] = extinguish_fire(env, trying_extinguish)

    # - Communication
    if len(trying_communicate) > 0:
        p = ProtocolHandler()
        network = None
        p.check_communication(trying_communicate,network)

    # 3. Updating state
    for agent in env.components['agents']:
        if agent.resources['battery'] > 0:
            if 0 < positions[agent.index][0] < env.dim[0]\
            and 0 < positions[agent.index][1] < env.dim[1]:
                agent.position = positions[agent.index]
                agent.direction = directions[agent.index]
                agent.resources['battery'] -= costs[agent.index]['battery']
                agent.resources['water'] -= costs[agent.index]['water']

                info['costs_reward'] = 0
            else:
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
            agent.resources['water'] = 0
    
    env.update()
    next_state = env.copy()
    return next_state, info

def do_policy_action(actions, env):
    controler_actions = {}
    for agent in env.components['agents']:
        if isinstance(actions,dict):
            if agent.index not in actions:
                agent_action = 0
            else:
                agent_action = actions[agent.index]
        else:
            agent_action = actions
        
        if agent_action == 0 or\
        agent_action == None: 
            controler_actions[agent.index ] = 0
        elif agent_action == 1: #extinguisher
            policy = env.import_method('extinguisher')

            copied_env = env.change_action_mode(new_mode='control')
            copied_env.perspective = agent.index
            observable_env = copied_env.get_observation()

            controler_actions[agent.index], agent.target = policy(observable_env, agent)
            env.targets[agent.index] = agent.target
        elif agent_action == 2: #explorer
            policy = env.import_method('explorer')

            copied_env = env.change_action_mode(new_mode='control')
            copied_env.perspective = agent.index
            observable_env = copied_env.get_observation()
            
            controler_actions[agent.index ], agent.target = policy(observable_env, agent)
            env.targets[agent.index] = agent.target
        else:
            raise NotImplemented

    next_state, info = do_control_action(controler_actions,env)
    return next_state, info

def extinguish_fire(env, trying_extinguish):
    global FIRE_UPDATE, FIRE_EXTINGUISH_DISTANCE
    extinguish_reward = 0

    # 1. Calculating the work impact
    wt = {}
    for agent in trying_extinguish:
        extinguished_one = False
        for fire in env.components['fire']:
            if not fire.extinguished and is_visible(fire.position,agent.position,\
             agent.direction,agent.radius,agent.angle):
                distance = np.linalg.norm(np.array(agent.position) - \
                                            np.array(fire.position))
                                        
                if distance <= FIRE_EXTINGUISH_DISTANCE and\
                agent.resources['water'] > 0 and agent.resources['battery'] > 0:
                    extinguished_one = True
                    if fire.position not in wt:
                        wt[fire.position] = 1
                    else:
                        wt[fire.position] += 1
            if extinguished_one:
                break

    updated = False
    for fire in env.components['fire']:
        if fire.position in wt:
            extinguished, expanded_fire = fire.update(wt[fire.position])
            extinguish_reward += 1
        elif time() - env.last_fire_update > FIRE_UPDATE:
            extinguished, expanded_fire = fire.update(0)
            updated = True
        else:
            expanded_fire = False
            extinguished = False

        if expanded_fire:
            expand_fire(env, fire.position)
        if extinguished:   
            extinguish_reward += 10

    if updated:      
        env.last_fire_update = time()
    
    return extinguish_reward

def expand_fire(real_env, ref_fire):
    x = ref_fire[0] + rd.randint(-1,1)*5
    y = ref_fire[1] + rd.randint(-1,1)*5

    # adjusting the position to the current env dims
    if x > real_env.dim[0]:
        x = real_env.dim[0] - 1
    if x < 0:
        x = real_env.dim[0] + 1

        
    if y > real_env.dim[1]:
        y = real_env.dim[1] - 1
    if y < 0:
        y = real_env.dim[1] + 1

    # expanding fire
    real_env.components['fire'].append(Fire(
        position=(x,y),
        level = 1 ))

def spawn_fire(real_env):
    global MAX_SIMULTANEOUS_FIRE, SPAWN_DELAY, SPAWN_PROBABILITY
    
    # checking the spawn conditions
    spawn_allowed = ((time() - real_env.last_spawn) > SPAWN_DELAY)
    if spawn_allowed:
        # counting the current number of fires (not extinguished) in the environment
        fires_in_the_env = sum([not f.extinguished for f in real_env.components['fire']])
        # checking the max number of spawns possible
        n_spawns =  MAX_SIMULTANEOUS_FIRE - fires_in_the_env \
         if MAX_SIMULTANEOUS_FIRE - fires_in_the_env > 0 else 0
        # spawning fires
        for i in range(n_spawns):
            coin = rd.uniform(0,1)
            if coin < SPAWN_PROBABILITY:
                real_env.components['fire'].append(Fire(
                    position=(rd.randint(real_env.pad,real_env.dim[0] - real_env.pad),
                        rd.randint(real_env.pad,real_env.dim[1] - real_env.pad)),
                    level = rd.randint(1, 2) ))

        real_env.last_spawn = time()









"""
3. ENVIRONMENT MODEL METHODS
"""
def end_condition(state):
    # no end condition
    return False

def smartfirebrigade_transition(actions, real_env):
    # Fire spawn
    # - If spawn_delay is None, fire will not spawn in the environment
    # hence, the initial set of fires is the set of tasks to be accomplished
    # - Else, new fires will spawn
    global SPAWN_DELAY
    if SPAWN_DELAY is not None:
        spawn_fire(real_env)

    # Checking agents target
    for i in range(len(real_env.components['agents'])):
        real_env.components['agents'][i].target = real_env.targets[real_env.components['agents'][i].index]\
            if real_env.components['agents'][i].index in real_env.targets else None

    # Updating agents
    if real_env.action_mode == 'control':
        next_state, info = do_control_action(actions, real_env)
    elif real_env.action_mode == 'policy':
        next_state, info = do_policy_action(actions, real_env)
    else:
        raise NotImplemented

    # Updating agents' memory
    for agent in real_env.components['agents']:
        copied_env = real_env.copy()
        copied_env.perspective = agent.index
        obs_env = copied_env.get_observation()
        agent.update_memory(obs_env)

    # retuning the results
    return next_state, info

# The reward must keep be calculated keeping the partial observability in mind
def reward(state, next_state):
    return 0

# Changes the actual environment to partial observed environment
def environment_transformation(copied_env):
    # 1. Getting the state in the ad hoc agent perspective
    agent = None
    for a in copied_env.components['agents']:
        if a.index == copied_env.perspective:
            agent = a
            break

    if agent is not None:
        ###
        # BUILDING OBSERVABLE STATE
        ###
        copied_env.state = {}
        # 1. Removing the invisible agents and tasks from environment
        invisible_agents = []
        for i in range(len(copied_env.components['agents'])):
            if not is_visible(copied_env.components['agents'][i].position,
            agent.position, agent.direction, agent.radius, agent.angle) and \
            copied_env.components['agents'][i] != agent:
                invisible_agents.append(i)

        invisible_tasks = []
        for i in range(len(copied_env.components['fire'])):
            if not is_visible(copied_env.components['fire'][i].position,
            agent.position, agent.direction, agent.radius, agent.angle) or \
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
            elif i in invisible_tasks and copied_env.components['agents'][i].position in agent.memory['agents']:
                pos = copied_env.components['agents'][i].position
                copied_env.state['agent'].append(pos)
            elif copied_env.components['agents'][i].index != agent.index:
                del copied_env.components['agents'][i]

        # - tasks
        for i in range(len(copied_env.components['fire'])-1, -1,-1):
            # visible
            if i not in invisible_tasks and not copied_env.components['fire'][i].extinguished:
                pos = copied_env.components['fire'][i].position
                copied_env.state['fire'].append(pos)
            # memory
            elif i in invisible_tasks and copied_env.components['fire'][i].position in agent.memory['fire']:
                pos = copied_env.components['fire'][i].position
                copied_env.state['fire'].append(pos)
            # not visible
            else:
                del copied_env.components['fire'][i]

    return copied_env












"""
    SmartFireBrigade Environment
"""
class FireBrigadeState(Space, object):
    
    def __init__(self):
        # Initialising the state space
        super(FireBrigadeState, self).__init__(dtype=dict)

    def sample(self, seed=None):
        state = {'agents':[],'fire':[]}
        return state

class SmartFireBrigadeEnv(AdhocReasoningEnv):
    move_length = 1

    ###
    # ENVIRONMENT
    ###
    def __init__(self, components, dim, action_mode='control',display=False):
        # Single-Agent Perspective Markovian Model
        self.perspective = None
        self.action_mode = action_mode
        self.extinguish_distance = 10

        # State Set
        state_set = StateSet(FireBrigadeState(), end_condition)     

        # Actions
        self.actions = {}
        self.actions_count = {}
        self.targets = {}
        if action_mode == 'control':
            self.action_dict = {
                0:'NOOP',
                1:'NORTH',
                2:'EAST',
                3:'SOUTH',
                4:'WEST',
                5:'EXTINGUISH',
                6:'COMMUNICATE'
            }
            action_space = spaces.Discrete(6)
        elif action_mode == 'policy':
            self.action_dict = {
                0:'NOOP',
                1:'EXTINGUISHER',
                2:'EXPLORER'
            }
            action_space = spaces.Discrete(3)
        else:
            print(action_mode)
            raise NotImplemented
        self.action_list = range(len(self.action_dict))

        # Transition function
        transition_function = smartfirebrigade_transition

        # Reward function
        reward_function = reward

        # Observation function
        observation_space = environment_transformation

        # Rendering variables
        self.viewer = None
        self.display = display

        # Experiment variables
        self.start_time = time()
        self.last_fire_update = time()
        self.last_spawn = time()

        # Rendering variables
        self.move_scale = 0.03
        self.rotation_scale = 0.005

        # Defining the Env parameters
        self.dim = dim
        self.square_meters = (dim[0]*dim[1])
        self.move_cost = 0.01 
        self.extinguish_battery = 0.02 
        self.extinguish_water = 0.2 

        # Initialising the parallel variables
        self.threads = {}

        # Initialising the Adhoc Reasoning Env
        super(SmartFireBrigadeEnv, self).__init__(state_set, transition_function, 
            action_space, reward_function, observation_space, components)
        self.state_set.initial_components = self.copy_components(components)

        # Setting the inital state
        self.state_set.initial_state = {'agents':[],'fire':[]}
        for agent in self.state_set.initial_components['agents']:
            self.state_set.initial_state['agents'].append(agent.position)
        for fire in self.state_set.initial_components['fire']:
            self.state_set.initial_state['fire'].append(fire.position)

    def import_method(self, agent_type):
        try:
            module = import_module('src.reasoning.smartfirebrigade.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = SmartFireBrigadeEnv(components, dim=self.dim, action_mode=self.action_mode,display=self.display)
        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode
        copied_env.perspective = self.perspective
        copied_env.last_fire_update = self.last_fire_update

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = self.copy_state()

        return copied_env

    def change_action_mode(self,new_mode):
        if new_mode == self.action_mode:
            copied_env = self.copy()
        else:
            components = self.copy_components(self.components)
            copied_env = SmartFireBrigadeEnv(components, dim=self.dim, action_mode=new_mode,display=self.display)
            copied_env.viewer = self.viewer
            copied_env.display = self.display
            copied_env.episode = self.episode
            copied_env.perspective = self.perspective

            # Setting the initial state
            copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
            copied_env.state = self.copy_state()
        return copied_env

    def copy_state(self):
        copied_state = {'agents':[],'fire':[]}
        for x in self.state['agents']:
            copied_state['agents'].append(x)
        for x in self.state['fire']:
            copied_state['fire'].append(x)
        return copied_state
    
    def update(self):
        self.state = {'agents':[],'fire':[]}
        for agent in self.components['agents']:
            self.state['agents'].append(agent.position)
        for fire in self.components['fire']:
            if not fire.extinguished:
                self.state['fire'].append(fire.position)
    
    def threading_f(self,agent):
        copied_env = self.copy()
        method = copied_env.import_method(agent.type)

        copied_env.perspective = agent.index
        observable_env = copied_env.get_observation()
        new_action, target = method(observable_env, agent)
        
        with open('./tmp/'+agent.index+'.txt','w') as action_file:
            if new_action is not None:
                action_file.write(str(new_action)+',')
            else:
                action_file.write(str(None)+',')
            if target is not None:
                action_file.write(str(target[0])+','+str(target[1]))
            else:
                action_file.write(str(None)+','+str(None))

    def get_step_actions(self):
        global POLICY_MAX_STEPS
        for agent in self.components['agents']:
            if not os.path.exists('./tmp/'+str(agent.index)+'.txt'):
                if agent.index not in self.threads:
                    thread = Process(target=self.threading_f, args=(agent,))
                    self.threads[agent.index] = thread
                    self.threads[agent.index].start()
                elif not self.threads[agent.index].is_alive():
                    thread = Process(target=self.threading_f, args=(agent,))
                    self.threads[agent.index] = thread
                    self.threads[agent.index].start()
                elif self.action_mode == 'control':
                    self.actions[agent.index] = 0
                elif self.action_mode == 'policy':
                    if agent.index in self.actions_count:
                        if self.actions_count[agent.index] == POLICY_MAX_STEPS:
                            self.actions[agent.index] = 0
                            self.actions_count[agent.index] = 0
                        else:
                            self.actions_count[agent.index] += 1
                    else:
                        self.actions_count[agent.index] = 0
                else:
                    raise NotImplemented

            else:
                with open('./tmp/'+agent.index+'.txt','r') as action_file:
                    for line in action_file:
                        token = line.split(',')
                        self.actions[agent.index] = int(token[0]) if token[0] != 'None' else 0
                        self.targets[agent.index] = None if token[1] == 'None' else (int(token[1]),int(token[2]))
                    
                    self.actions_count[agent.index] = 0 if agent.index not in self.actions_count\
                                                            else self.actions_count[agent.index] + 1
                if agent.index in self.actions_count and self.actions_count[agent.index] > 1:
                    os.remove('./tmp/'+agent.index+'.txt')

        return self.actions

    def get_actions_dict(self):
        return self.action_dict

    def get_actions_list(self):
        return self.action_list

    def get_action(self, name):
        for key in self.action_dict:
            if self.action_dict[key] == name:
                return key
        return None

    def get_fire(self, position):
        for fire in self.components['fire']:
            if position[0] == fire.position[0] and\
                position[1] == fire.position[1] and not fire.extinguished: 
                return fire
        return None

    def get_agent(self,position):
        for ag in self.components['agents']:
            if position[0] == ag.position[0] and\
                position[1] == ag.position[1]: 
                return ag
        return None


    def check_position(self,position):
        agent = self.get_agent(position)
        fire = self.get_fire(position)
        return(agent,fire)

    def state_is_equal(self, state):
        for x in state['agents']:
            if x not in self.state['agents']:
                return False
        for x in state['fire']:
            if x not in self.state['fire']:
                return False
        return True

    def observation_is_equal(self, obs):
        # 1. comparing state agents
        for agent1 in self.state['agents']:
            for agent2 in obs.state['agents']:
                if agent1 != agent2:
                    return False

        # 2. comparing state fires
        for fire1 in self.state['fire']:
            for fire2 in obs.state['fire']:
                if fire1 != fire2:
                    return False
        return True

    def sample_state(self,agent):
        global N_FIRE_SAMPLE
        # spawn random reward spot in the environment
        copied_env = self.copy()
        copied_env.perspective = agent.index
        observable_env = copied_env.get_observation()

        for i in range(N_FIRE_SAMPLE):
            pos = (rd.randint(0,self.dim[0]),rd.randint(0,self.dim[1]))
            if not is_visible(pos,agent.position,agent.direction,agent.radius,agent.angle)\
            and np.linalg.norm(np.array(pos) - np.array(agent.position)) > self.extinguish_distance:
                fire = Fire(pos,level=3.0)
                observable_env.components['fire'].append(fire)
                observable_env.state['fire'].append(pos)

        return observable_env

    ###
    # RENDERING
    ###
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
            self.screen_width, self.screen_height, self.pad = 800, 800,10
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

        else:
            ####
            ##
            #   Non-smooth transitions
            ## 
            ####
            # bettery and water level
            for i in range(len(self.components['agents'])):
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

            ####
            ##
            #   Smooth transition
            ##
            ####
            TOTAL_FRAMES = 30
            initial_dir = [self.d_agents[i].rotation for i in range(len(self.components['agents']))] 
            initial_pos = [((self.d_agents[i].translation[0] - self.pad)/self.scale,
                            (self.d_agents[i].translation[1] - self.pad)/self.scale)\
                             for i in range(len(self.components['agents']))] 
            initial_scale = [self.d_fire[i].attrs[-1].scale for i in range(len(self.d_fire))] 
            for frame in range(TOTAL_FRAMES):   
                for i in range(len(self.components['agents'])):
                    direc = float((frame/TOTAL_FRAMES)*(self.components['agents'][i].direction - initial_dir[i])) + initial_dir[i]
                    pos = (
                        ((frame+1)/TOTAL_FRAMES)*(self.components['agents'][i].position[0] - initial_pos[i][0]) + initial_pos[i][0],
                        ((frame+1)/TOTAL_FRAMES)*(self.components['agents'][i].position[1] - initial_pos[i][1]) + initial_pos[i][1]
                        )
                    self.d_agents[i].set_translation(0,0)
                    self.d_agents[i].set_rotation(direc)
                    self.d_agents[i].set_translation(
                        self.scale*pos[0] + self.pad,
                        self.scale*pos[1] + self.pad) 
            
                # checking the fire
                for i in range(len(self.components['fire'])):
                    if i >= len(self.d_fire):
                        fire = self.draw_fire(self.components['fire'][i].position)
                        self.viewer.add_geom(fire)
                        self.d_fire.append(fire)
                        initial_scale.append((self.components['fire'][i].spreading_level,self.components['fire'][i].spreading_level))

                    elif self.components['fire'][i].extinguished:
                        self.d_fire[i].attrs[-1].set_scale(0.0, 0.0)

                    else:
                        position = self.components['fire'][i].position 
                        scale = ((frame+1)/TOTAL_FRAMES)*\
                            (self.components['fire'][i].spreading_level - initial_scale[i][0]) + initial_scale[i][0]
                        self.d_fire[i].attrs[-1].set_translation(0,0)
                        self.d_fire[i].attrs[-1].set_scale(scale, scale)
                        self.d_fire[i].attrs[-1].set_translation(\
                            self.scale*position[0] + self.pad,
                            self.scale*position[1] + self.pad)

                # timer
                self.timer = self.draw_timer()
                self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
        execution_time = (time() - self.start_time)
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
