from copy import deepcopy
from gym import spaces
import numpy as np
import random 
import os

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

SLIP_P = 0.
MISS_OBS = 0.15

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display=False):
    scenario, scenario_id = load_default_scenario_components(scenario_id)

    scenario_dim = (int(scenario["dim"][0]),int(scenario["dim"][1]))
    agent = Agent(index= 0, type= method)
    components = {"agents":[agent], "black":scenario["black"]}
    agent_position = (int(scenario_dim[0]/2),int(scenario_dim[1]/2))

    env = MazeEnv(agent_position,scenario_dim,components=components,display=display)
    return env, scenario_id

def load_default_scenario_components(scenario_id):
    if scenario_id >= 4:
        print('There is no default scenario with id '+str(scenario_id)+' for the Maze problem. Setting scenario_id to 0.')
        scenario_id = 0

    default_scenarios_components = [{"dim":[5,5],"black":[(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(3,4),(4,4)]},
            {"dim":[8,8],"black":[(0,1),(0,3),(0,5),(0,7),(2,1),(2,3),(2,5),(2,7),(6,1),(6,3),(6,5),(6,7),(4,1),(4,5),(4,7)]},
            {"dim":[6,6],"black":[(1,1),(1,4),(4,1)]},
            {"dim":[3,3],"black":[(0,0),(1,0),(2,0),(1,1),(2,1),(2,2)]}]

    return default_scenarios_components[scenario_id], scenario_id

"""
    Support classes
"""
class MazeEnvState(spaces.Space):
    def __init__(self,n):
        super(MazeEnvState,self).__init__(dtype=dict) 

    def sample(self,seed=None):
        raise NotImplemented

class Agent(AdhocAgent):
    def __init__(self,index,type="random"):
        super(Agent,self).__init__(index,type)
        self.type = type
        self.index = index
    
    def copy(self):
        copy_agent = Agent(self.index,self.type)
        return copy_agent

def end_condition(state):
    return (state.state['belief']>0.999).any()

def do_action(env):
    action = env.components['agents'][0].next_action
    # 1. Performing state transition
    if env.simulation:
        new_pos = None
        x, y = env.state['agent']
        if env.action_dict[action] == 'EAST':
            new_pos = ((x+1)%env.dim[0],y)
        elif env.action_dict[action] == 'WEST':
            new_pos = ((x-1)%env.dim[0],y)
        elif env.action_dict[action] == 'NORTH':
            new_pos = (x,(y+1)%env.dim[1])
        elif env.action_dict[action] == 'SOUTH':
            new_pos = (x,(y-1)%env.dim[1])
        else:
            new_pos = (x,y)
        env.state['agent'] = new_pos

    # 2. Updating real position
    else:
        new_pos = None
        x, y = env.agent_position
        if env.action_dict[action] == 'EAST':
            new_pos = ((x+1)%env.dim[0],y)
        elif env.action_dict[action] == 'WEST':
            new_pos = ((x-1)%env.dim[0],y)
        elif env.action_dict[action] == 'NORTH':
            new_pos = (x,(y+1)%env.dim[1])
        elif env.action_dict[action] == 'SOUTH':
            new_pos = (x,(y-1)%env.dim[1])
        else:
            new_pos = (x,y)
        env.agent_position = new_pos

    # 3. Performing checking action (if it was taken)
    obs = None
    if env.action_dict[action] == 'CHECK':
        global MISS_OBS
        if env.simulation:
            if random.random() >= MISS_OBS:
                obs = env.grid[env.state['agent']]
            else:
                obs = 0 if env.grid[env.state['agent']] == 1 else 1
        else:
            if random.random() >= MISS_OBS:
                obs = env.grid[env.agent_position]
            else:
                obs = 0 if env.grid[env.agent_position] == 1 else 1
            
    env.state['obs'] = obs
    env.state['belief'] = belief_update(env,action)

    return env, {}

def belief_update(env,prev_action):
    grid = env.grid
    obs = env.state['obs']
    belief = env.state['belief']

    if np.sum(belief)==0:
        return np.ones_like(belief)

    posterior = np.zeros_like(belief)
    steps = [(1,0),(-1,0),(0,1),(0,-1)]
    for x in range(0,grid.shape[0]):
        for y in range(0,grid.shape[1]):
            if prev_action < 4:
                next_state = ((x+steps[prev_action][0])%grid.shape[0],(y+steps[prev_action][1])%grid.shape[1])
                posterior[next_state] += belief[(x,y)]
            else:  
                if grid[(x,y)]==obs:    
                    posterior[(x,y)] = (belief[(x,y)] *(1 - MISS_OBS))/(belief[(x,y)]*(1-MISS_OBS) + (1-belief[(x,y)])*MISS_OBS)
                else:
                    posterior[(x,y)] = (belief[(x,y)]*(MISS_OBS))/(belief[(x,y)]*MISS_OBS + (1-belief[(x,y)])*(1-MISS_OBS))
                    
    posterior /= sum(sum(posterior))
    return posterior

def maze_transition(action,real_env):
    real_env.components['agents'][0].next_action = action
    next_state,info = do_action(real_env) 
    return next_state,info

def reward(state,next_state):
    return -entropy(next_state.state['belief']) + entropy(state['belief']) 

def entropy(arr):
    if arr is None or not 0.99 <= np.sum(arr) <=1.1 :
        return -10
    s = 0
    for row in arr:
        for p in row:
            s -= np.log(p+1e-7)*p     
    return s

def environment_transformation(copied_env):
    return copied_env

class MazeEnv(AdhocReasoningEnv):

    action_dict = {\
        0:'EAST',
        1:'WEST',
        2:'NORTH',
        3:'SOUTH',
        4:'CHECK'
    }

    def __init__(self,agent_pos, dim, components,display=False):
        ###
        # Env Settings
        ###
        self.dim = dim
        
        state_set = StateSet(MazeEnvState(dim),end_condition)
        transition_function = maze_transition
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(MazeEnv,self).__init__(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)
        self.state_set.initial_components = self.copy_components(components)

        belief = self.initialise_belief()
        self.state_set.initial_state = {'agent':None, 'belief':belief, 'obs':None}
        self.state = {'agent':None, 'belief':belief, 'obs':None}

        self.grid = self.initialise_grid()
        self.agent_position = agent_pos

        ###
        # Setting graphical interface
        ###self.screen = None
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

    def show_belief(self):
        for x in range(self.state['belief'].shape[0]):
            for y in reversed(range(self.state['belief'].shape[1])):
                print("{:.2f}".format(self.state['belief'][x,y]),end='\t')
            print()

    def initialise_grid(self):
        grid = np.zeros(self.dim)
        for pos in self.components['black']:
            grid[pos] = 1
        return grid

    def initialise_belief(self):
        belief = np.ones(self.dim)
        belief /= sum(sum(belief))
        return belief

    def get_all_states(self):
        all_states = []
        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                all_states.append((x,y))
        return all_states

    def get_trans_p(self,action):
        return [self,1]

    def get_obs_p(self,action):
        return [self,1]

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
        copied_env = MazeEnv(self.agent_position,self.dim,components,self.display)
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.renderer = self.renderer
        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env
    
    # The environment is partially observable by definition
    def state_is_equal(self,state):         
        return (state.state['agent']==self.state['agent'])

    def observation_is_equal(self,obs):
        return (obs.state['obs']==self.state['obs'])

    def sample_state(self,agent):
        u_env = self.copy()
        obs = u_env.get_observation()

        possible_positions = []
        for pos in self.get_all_states():
            if obs.grid[pos] == obs.state['obs'] or obs.state['obs'] is None:
                possible_positions.append(pos)

        obs.state['agent'] = random.choice(possible_positions)
        return obs
    
    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def get_actions_list(self):
        return [i for i in range(len(self.action_dict))]

    def get_adhoc_agent(self):
        return self.components['agents'][0]

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

        # background
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill(self.colors['white'])
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        # grid
        grid_width, grid_height = 700, 700
        self.grid_surf = pygame.Surface((grid_width, grid_height))
        self.grid_surf.fill(self.colors['white'])
        
        for pos in self.components['black']:
            bc_x, bc_y = pos[0]*(grid_width/self.dim[0]),pos[1]*(grid_height/self.dim[1])
            black_cell = pygame.Rect((bc_x,bc_y),(int(grid_width/self.dim[0]),int(grid_height/self.dim[1])))
            pygame.gfxdraw.box(self.grid_surf,black_cell,self.colors['black'])

        for column in range(-1,self.dim[1]):
            pygame.draw.line(self.grid_surf,self.colors['darkgrey'],
                                (0*grid_width,(column+1)*(grid_height/self.dim[1])),
                                (1*grid_width,(column+1)*(grid_height/self.dim[1])),
                                int(0.1*np.sqrt((grid_width/self.dim[0])*(grid_height/self.dim[1]))))
        for row in range(-1,self.dim[0]):
            pygame.draw.line(self.grid_surf,self.colors['darkgrey'],
                                ((row+1)*(grid_width/self.dim[0]),0*grid_height),
                                ((row+1)*(grid_width/self.dim[0]),1*grid_height),
                                int(0.1*np.sqrt((grid_width/self.dim[0])*(grid_height/self.dim[1]))))

        ###
        # Agent
        ###
        x = int(self.agent_position[0]*(grid_width/self.dim[0]) + 0.5*(grid_width/self.dim[0]))
        y = int(self.agent_position[1]*(grid_height/self.dim[1]) + 0.7*(grid_height/self.dim[1]))
        r = int(0.3*np.sqrt((grid_width/self.dim[0])*(grid_height/self.dim[1])))
        if self.components['agents'][0].next_action == 4:
            gfxdraw.filled_circle(self.grid_surf,x,y,r,self.colors['green'])
        else:
            gfxdraw.filled_circle(self.grid_surf,x,y,r,self.colors['red'])
        
        ###
        # Belief
        ###
        myfont = pygame.font.SysFont("Ariel", int(0.4*np.sqrt((grid_width/self.dim[0])*(grid_height/self.dim[1]))))
        for row in range(self.dim[0]):
            for col in range(self.dim[1]):
                if self.state['belief'][row,col] < 0.001:
                    rx = int(row*(grid_width/self.dim[0]) + 0.1*(grid_width/self.dim[0]))
                    ry = int(col*(grid_height/self.dim[1]) + 0.1*(grid_height/self.dim[1]))
                    label = myfont.render("%.3f"%(self.state['belief'][row,col]), True, self.colors['red'])
                    label =  pygame.transform.flip(label, False, True)
                    self.grid_surf.blit(label, (rx, ry))
                else:
                    rx = int(row*(grid_width/self.dim[0]) + 0.1*(grid_width/self.dim[0]))
                    ry = int(col*(grid_height/self.dim[1]) + 0.1*(grid_height/self.dim[1]))
                    label = myfont.render("%.3f"%(self.state['belief'][row,col]), True, self.colors['green'])
                    label =  pygame.transform.flip(label, False, True)
                    self.grid_surf.blit(label, (rx, ry))

        ##
        # Text
        ##
        act = self.action_dict[self.components['agents'][0].next_action] \
            if self.components['agents'][0].next_action is not None else None
        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Episode "+str(self.episode) + \
            " | Action: "+str(act), True, self.colors['black'])
        self.screen.blit(label, (10, 10))

        ##
        # Displaying
        ##
        self.grid_surf = pygame.transform.flip(self.grid_surf, False, True)
        self.screen.blit(self.grid_surf, (50, 50))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )