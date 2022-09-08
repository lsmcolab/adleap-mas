from gym import spaces
import numpy as np
import random as rd
import os

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

# Implementation based on "Heuristic Search Value Iteration
# for POMDP" : https://arxiv.org/pdf/1207.4166.pdf

"""
    Load Scenario method
"""
def load_default_scenario(method,scenario_id=0,display=False):
    scenario, scenario_id = load_default_scenario_components(scenario_id)

    scenario_dim = int(scenario["dim"])
    agent = Agent(index= 0, position= (0,0), type= method)
    components = {"rocks":scenario['rocks'],"agents":[agent]}
    
    env = RockSampleEnv(components=components,dim=scenario_dim,display=display)
    return env, scenario_id

def load_default_scenario_components(scenario_id):
    if scenario_id >= 4:
        print('There is no default scenario with id '+str(scenario_id)+' for the RockSample problem. Setting scenario_id to 0.')
        scenario_id = 0

    default_scenarios_components = [\
        {"dim":10,"rocks":[\
            Rock(0,(2,2),"Bad"),Rock(1,(5,2),"Good"),Rock(2,(3,7),"Good"),Rock(3,(6,8),"Bad")]},
        {"dim":10,"rocks":[\
            Rock(0,(2,2),"Good"),Rock(1,(5,2),"Good"),Rock(2,(3,7),"Good"),Rock(3,(6,8),"Good")]},
        {"dim":10,"rocks":[\
            Rock(0,(2,2),"Bad"),Rock(1,(4,2),"Good"),Rock(2,(6,2),"Good"),Rock(3,(8,2),"Bad"),\
            Rock(4,(2,8),"Good"),Rock(5,(4,8),"Bad"),Rock(6,(6,8),"Good"),Rock(7,(8,8),"Bad")]},
        {"dim":10,"rocks":[\
            Rock(0,(2,3),"Bad"),Rock(1,(4,3),"Bad"),Rock(2,(6,3),"Bad"),Rock(3,(8,3),"Bad"),\
            Rock(4,(2,8),"Good"),Rock(5,(4,8),"Bad"),Rock(6,(6,8),"Bad"),Rock(7,(8,8),"Bad")]}]

    return default_scenarios_components[scenario_id], scenario_id

"""
    Support classes
"""
class Agent(AdhocAgent):

    def __init__(self,index,position,type="random"):
        super(Agent,self).__init__(index,type)
        self.position = position
        self.type = type

    def copy(self):
        copy_agent = Agent(self.index,self.position,self.type)
        return copy_agent

class Rock():

    # Use cells=None to denote that it is rectangle
    def __init__(self,index,position,rtype,belief=0.5):
        self.index = index
        self.position = np.array(position)
        self.type = rtype
        self.belief = belief # P(Rock=Good|H_t)

    def copy(self):
        copy_rock = Rock(self.index,self.position,self.type,self.belief)
        return copy_rock

    def check(self,pos):
        eta = np.exp(-0.2*distance(self.position,pos))
        prob = eta + (1-eta)*0.5

        # Bayesian Update of Belief
        # P(Rock=Good|Obs,H_t) = 
        #   P(Obs|Rock=Good,H_t)*P(Rock=Good,H_t)/P(Obs|H_t)
        # P(Obs|H_t) = P(Obs|Rock=Good)*P(Rock=Good|H_t) + P(Obs|Rock=Bad)*P(Rock=Bad|H_t)

        obs = "None"
        if rd.random() < prob:
            if self.type=="Good":
                obs = "Good"
            else:
                obs = "Bad"
        else:
            if self.type == "Good":
                obs = "Bad"
            else:
                obs = "Good" 

        if obs=="Good":
            belief = self.belief*prob/(self.belief*prob + (1-self.belief)*(1-prob))
        else:
            belief = self.belief*(1-prob)/(self.belief*(1-prob)+(1-self.belief)*prob)   
        self.belief = belief

        return obs, belief


def end_condition(state):
    return (state.state['agent'][0] == (state.dim-1))\
        and (state.state['agent'][1] == (state.dim-1))

def distance(pos_1,pos_2):
    return np.linalg.norm(pos_1-pos_2)

def do_action(env,action):
    info = {}

    # Getting info
    dim = env.dim
    old_pos = env.state['agent']

    x,y = old_pos[0], old_pos[1]
    new_pos = np.array([x,y])

    if env.action_dict[action] == 'East':
        if (x+1) < dim:
            new_pos = np.array([x+1,y])
        else:
            new_pos = np.array([-1,-1])
    elif env.action_dict[action] == 'West':
        new_pos = np.array([max(0,x-1),y])
    elif env.action_dict[action] == 'North':
        new_pos = np.array([x,min(y+1,dim-1)])
    elif env.action_dict[action] == 'South':
        new_pos = np.array([x,max(0,y-1)])
    elif env.action_dict[action] == 'Sample':
        for i in range(len(env.components['rocks'])):
            if (env.components['rocks'][i].position == new_pos).all():
                if env.components['rocks'][i].type == "Good":
                    if env.simulation:
                        info['reward'] = 100*env.state['beliefs'][i]
                    else:
                        info['reward'] = 100
                    env.state['obs'][i] = "Bad"
                    env.components['rocks'][i].type = "Bad"
                    env.state['beliefs'][i] = 0.0
                    env.components['rocks'][i].belief = 0.0
                else:
                    if env.simulation:
                        info['reward'] = -100*(1-env.state['beliefs'][i])
                    else:
                        info['reward'] = -100
                    env.state['obs'][i] = "Bad"
                    env.state['beliefs'][i] = 0.0
                    env.components['rocks'][i].belief = 0.0
                break
    else: # Sense-
        rock_index = action-len(env.action_dict)
        rock = env.components['rocks'][rock_index]
        obs, belief = rock.check(new_pos)
        env.state['obs'][rock_index] = obs
        info['sense reward'] = abs(belief - env.state['beliefs'][rock_index])
        env.state['beliefs'][rock_index] = belief

    env.state['agent'] = new_pos
    env.components['agents'][0].position = new_pos

    return env,info

def rocksample_transition(action,real_env):
    real_env.components['agents'][0].next_action = action
    next_state,info = do_action(real_env,action)
    return next_state,info

def reward(state,next_state):
    if end_condition(next_state):
        return 10
    return 0

def environment_transformation(copied_env):
    return copied_env


class RockSampleState(spaces.Space):    
    def __init__(self,n):
        super(RockSampleState,self).__init__(shape=(n,n),dtype=np.float64)
    
    def sample(sef,seed=None):
        state = {'agent' : [], 'rocks':[], 'beliefs':[], 'obs':[]}
        return state

class RockSampleEnv(AdhocReasoningEnv):

    action_dict = {
        0 : 'East',
        1 : 'West',
        2 : 'North',
        3 : 'South',
        4 : 'Sample'    
    }

    observation_dict = {
        0: 'Good',
        1: 'Bad',
        2: 'None'
    }

    def __init__(self,components,dim,display=False):
        ###
        # Env Settings
        ###
        self.dim = dim
        self.no_of_rocks = len(components['rocks'])

        state_set = StateSet(RockSampleState(dim),end_condition)
        transition_function = rocksample_transition
        action_space = spaces.Discrete(len(self.action_dict)+self.no_of_rocks)
        reward_function = reward
        observation_space = environment_transformation
        
        # Adding the sense actions to the action dict
        for i in range(5,5+self.no_of_rocks):
            self.action_dict[i] = "Sense-{}".format(i-5)

        ###
        # Initialising the env
        ###
        super(RockSampleEnv,self).__init__(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)

        self.state_set.initial_components = self.copy_components(components)
        self.state_set.initial_state = {'agent':[],'rocks':[],'obs':[],'beliefs':[]}

        self.state_set.initial_state['agent'] = components['agents'][0].position
        
        for rock in components['rocks']:
            self.state_set.initial_state['rocks'].append(list(rock.position))
            self.state_set.initial_state['obs'].append('None')        
            self.state_set.initial_state['beliefs'].append(rock.belief)        

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
        copied_env = RockSampleEnv(components, self.dim,self.display)
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.renderer = self.renderer

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = self.copy_components(self.state)
        copied_env.simulation = self.simulation
        return copied_env

    def get_trans_p(self,action):
        return [self,1]
    
    def get_obs_p(self,action):
        if action < len(self.action_dict):
            return [self,1]
        else:
            if self.state['obs'][action-len(self.action_dict)] == 'Good':
                prob_good = self.state['beliefs'][action-len(self.action_dict)]
                return [self,prob_good]
            elif self.state['obs'][action-len(self.action_dict)] == 'Bad':
                prob_bad = 1 - self.state['beliefs'][action-len(self.action_dict)]
                return [self,prob_bad]
            else:
                return [self,0.5]
    
    # The environment is partially observable by definition
    def state_is_equal(self,state):    
        return self.state['agent'] == state.state['agent'] 

    def observation_is_equal(self,obs):
        return self.state['obs'] == obs.state['obs']

    def sample_state(self,agent):
        u_env = self.copy()
        obs = u_env.get_observation()

        for rock_index in range(len(obs.components['rocks'])):
            if np.random.random() < obs.state['beliefs'][rock_index]:
                obs.components['rocks'][rock_index].type = 'Good'
            else:
                obs.components['rocks'][rock_index].type = 'Bad'
        
        return obs
    
    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def get_actions_list(self):
        return [i for i in range(0,self.no_of_rocks+5)]

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

        self.screen_width, self.screen_height = 600, 800
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
        grid_width, grid_height = 500, 500
        self.grid_surf = pygame.Surface((grid_width, grid_height))
        grid_ret = pygame.Rect((0,0),(grid_width,grid_height))
        grid_image = pygame.image.load(os.path.abspath("./imgs/rocksample/space.png"))
        grid_image = pygame.transform.scale(grid_image, grid_ret.size)
        grid_image = grid_image.convert()
        self.grid_surf.blit(grid_image,grid_ret)
        for column in range(self.dim):
            pygame.draw.line(self.grid_surf,self.colors['white'],
                                (0*grid_width,(column+1)*(grid_height/self.dim)),
                                (1*grid_width,(column+1)*(grid_height/self.dim)))
        for row in range(self.dim):
            pygame.draw.line(self.grid_surf,self.colors['white'],
                                ((row+1)*(grid_width/self.dim),0*grid_height),
                                ((row+1)*(grid_width/self.dim),1*grid_height))

        # rock
        for rock in self.components['rocks']:
            rx, ry = rock.position[0]*(grid_width/self.dim),rock.position[1]*(grid_height/self.dim)

            grid_ret = pygame.Rect((rx,ry),(int(grid_width/self.dim),int(grid_height/self.dim)))
            if rock.type == "Bad":
                pygame.draw.rect(self.grid_surf,self.colors['red'],grid_ret)
            else:
                pygame.draw.rect(self.grid_surf,self.colors['green'],grid_ret)
                
            rock_ret = pygame.Rect((rx+int(0.05*grid_width/self.dim),ry+int(0.05*grid_height/self.dim)),\
                (int(0.9*grid_width/self.dim),int(0.9*grid_height/self.dim)))
            rock_img = pygame.image.load(os.path.abspath("./imgs/rocksample/rock.jpeg"))
            rock_img = pygame.transform.flip(rock_img,False,True)
            rock_img = pygame.transform.scale(rock_img, rock_ret.size)
            rock_img = rock_img.convert()
            self.grid_surf.blit(rock_img,rock_ret)

        # rover
        x, y = int(self.state['agent'][0]*(grid_width/self.dim)), int(self.state['agent'][1]*(grid_height/self.dim))
        rover_ret = pygame.Rect((x,y),(int(grid_width/self.dim),int(grid_height/self.dim)))
        rover_img = pygame.image.load(os.path.abspath("./imgs/rocksample/rover.png"))
        rover_img = pygame.transform.flip(rover_img,False,True)
        rover_img = pygame.transform.scale(rover_img, rover_ret.size)
        rover_img = rover_img.convert_alpha()
        rover_img.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGBA_MULT)
        self.grid_surf.blit(rover_img,rover_ret)

        ##
        # Action
        ##
        for agent in self.components['agents']:
            if agent.next_action > 4:
                rock = self.components['rocks'][agent.next_action-5]
                rx, ry = rock.position[0]*(grid_width/self.dim),rock.position[1]*(grid_height/self.dim)
                if self.state['obs'][agent.next_action-5] == 'Bad':
                    pygame.draw.line(self.grid_surf,self.colors['red'],
                                    (x+0.5*(grid_width/self.dim),y+0.5*(grid_height/self.dim)),
                                    (rx+0.5*(grid_width/self.dim),ry+0.5*(grid_height/self.dim)),
                                    int(0.1*np.sqrt((grid_width/self.dim)*(grid_height/self.dim))))
                else:
                    pygame.draw.line(self.grid_surf,self.colors['green'],
                                    (x+0.5*(grid_width/self.dim),y+0.5*(grid_height/self.dim)),
                                    (rx+0.5*(grid_width/self.dim),ry+0.5*(grid_height/self.dim)),
                                    int(0.1*np.sqrt((grid_width/self.dim)*(grid_height/self.dim))))

        ##
        # Text
        ##
        act = self.action_dict[self.components['agents'][0].next_action] \
            if self.components['agents'][0].next_action is not None else None
        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Episode "+str(self.episode) + \
            " | Action: "+str(act), True, self.colors['black'])
        self.screen.blit(label, (10, 10))

        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Agent position: ("+str(self.state['agent'][0])+","+str(self.state['agent'][1])+")",
                                         True, self.colors['black'])
        self.screen.blit(label, (10, 575))

        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Observation: ", True, self.colors['black'])
        self.screen.blit(label, (10, 625))
        for i in range(len(self.state['obs'])):
            if self.state['obs'][i] == 'Good':
                label = myfont.render("G (R"+str(i)+")", True, self.colors['green'])
                self.screen.blit(label, (200+(100*(i%4)), 625+(25*int(i/4))))
            elif self.state['obs'][i] == 'Bad':
                label = myfont.render("B (R"+str(i)+")", True, self.colors['red'])
                self.screen.blit(label, (200+(100*(i%4)), 625+(25*int(i/4))))
            else:
                label = myfont.render("N (R"+str(i)+")", True, self.colors['black'])
                self.screen.blit(label, (200+(100*(i%4)), 625+(25*int(i/4))))
                        
        myfont = pygame.font.SysFont("Ariel", 35)
        label = myfont.render("Belief: ", True, self.colors['black'])
        self.screen.blit(label, (10, 700))
        for i in range(len(self.state['beliefs'])):
            if self.state['beliefs'][i] > 0.5:
                label = myfont.render("%.2f (R%d)" % (self.state['beliefs'][i],i), True, self.colors['green'])
                self.screen.blit(label, (100+(125*(i%4)), 700+(25*int(i/4))))
            elif self.state['beliefs'][i] < 0.5:
                label = myfont.render("%.2f (R%d)" % (self.state['beliefs'][i],i), True, self.colors['red'])
                self.screen.blit(label, (100+(125*(i%4)), 700+(25*int(i/4))))
            else:
                label = myfont.render("%.2f (R%d)" % (self.state['beliefs'][i],i), True, self.colors['black'])
                self.screen.blit(label, (100+(125*(i%4)), 700+(25*int(i/4))))
        
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