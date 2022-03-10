from copy import deepcopy
from gym import spaces
import numpy as np
import random 

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet

class Agent(AdhocAgent):
    def __init__(self,index,position,type="random"):
        super(Agent,self).__init__(index,type)
        self.position = position
        self.type = type
        self.index = index
    
    def copy(self):
        copy_agent = Agent(self.index,self.position,self.type)
        return copy_agent

def end_condition(state):
    prob = state['belief']/np.sum(state['belief'])
    return (prob>0.999).any()

def update(env):
    env.state['agent'] = env.components['agents'][0].position
    return env.state

def do_action(env):
    action = env.components['agents'][0].next_action
    old_pos = env.components['agents'][0].position
    dim = env.dim
    x,y = old_pos[0],old_pos[1]
    new_pos = old_pos
    obs = None

    # EAST
    if action == 0:
        new_pos = ((x+1)%dim,y)
    # WEST
    elif action == 1:
        new_pos = ((x-1)%dim,y)
    # NORTH
    elif action == 2:
        new_pos = (x,(y+1)%dim)
    
    # SOUTH
    elif action == 3:
        new_pos = (x,(y-1)%dim)

    else:
        env.state['obs'] = env.grid[old_pos]
        obs = env.grid[old_pos]
    
    env.state['belief'] = belief_update(env.state['belief'],env.grid,obs,action)
    env.components['agents'][0].position = new_pos
    next_state = update(env)
    
    return next_state, {}

def belief_update(belief,grid,obs,prev_action):
    if np.sum(belief)==0:
        return belief

    posterior = np.zeros_like(belief)
    steps = [(1,0),(-1,0),(0,1),(0,-1)]
    n = grid.shape[0]
    for i in range(0,n):
        for j in range(0,n):
            if prev_action < 4:
                next_state = ((i+steps[prev_action][0])%n,(j+steps[prev_action][1])%n)
                posterior[next_state] += belief[(i,j)]
            else:
                if grid[(i,j)]==obs:    
                    posterior[(i,j)] = belief[(i,j)]

    return posterior

def maze_transition(action,real_env):
    real_env.components['agents'][0].next_action = action
    next_state,info = do_action(real_env) 
    return next_state,info

def reward(state,next_state):
    prob = state['belief']/np.sum(state['belief'])
    prob_next = next_state['belief']/np.sum(next_state['belief'])
    return -entropy(prob_next) + entropy(prob) 

def entropy(arr):
    if not 0.99 <= np.sum(arr)<=1.1 :
        return -10
    s = 0
    for row in arr:
        for p in row:
            s = s - np.log(p+1e-7)*p 
    return s

def environment_transformation(copied_env):
    copied_env.components['agents'][0].position = None
    copied_env.state['agent'] = None
    copied_env.episode += 1
    return copied_env

class MazeEnvState(spaces.Space):
    def __init__(self,n):
        super(MazeEnvState,self).__init__(shape=(n,n),dtype=np.float64) 

    def sample(self,seed=None):
        state = {'agent':None, 'belief':None, 'obs':None}
        return state

class MazeEnv(AdhocReasoningEnv):

    action_dict = {\
        0:'EAST',
        1:'WEST',
        2:'NORTH',
        3:'SOUTH',
        4:'CHECK'
    }

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

    def __init__(self,components,dim,display=False):
        assert ("agents" in components.keys() and "black" in components.keys())
        self.viewer = None
        self.display = display
        self.dim = dim
        state_set = StateSet(MazeEnvState(dim),end_condition)
        transition_function = maze_transition
        self.grid = np.zeros((dim,dim))
        action_space = spaces.Discrete(5)
        reward_function = reward
        observation_space = environment_transformation

        # Initialising the Env

        super(MazeEnv,self).__init__(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)

        self.state_set.initial_components = self.copy_components(components)
        self.state_set.initial_state = {'agent':None, 'belief':None, 'obs':None}
        self.state_set.initial_state['agent'] = components['agents'][0].position
        self.state_set.initial_state['belief'] = np.ones((dim,dim))
        self.state_set.initial_state['obs'] = None
        self.initialise_grid()

    def initialise_grid(self):
        for pos in self.components['black']:
            self.grid[pos] = 1
        return

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
        copied_env = MazeEnv(components, self.dim)
        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode
        copied_env.grid = deepcopy(self.grid)
        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env
    
    # The environment is partially observable by definition
    def state_is_equal(self,state):         
        return (state['obs']==self.state['obs'])

    def observation_is_equal(self,obs):
        state = obs.state
        return self.state_is_equal(state)

    def sample_state(self,agent,sample_p=0.1,n_sample=10):
        u_env = self.copy()
        obs = u_env.get_observation()
        vals = range(0,self.dim)
        obs.components['agents'][0].position = (random.sample(vals,1)[0],random.sample(vals,1)[0])
        obs.state['agent'] = obs.components['agents'][0].position
        return obs
    
    def sample_nstate(self, agent, n, sample_a=0.1, sample_t=0.1, n_sample=10):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent,sample_a,sample_t,n_sample))
        return sampled_states

    def get_actions_list(self):
        return [i for i in range(0,5)]

    def get_adhoc_agent(self):
        return self.components['agents'][0]

    def render(self,mode='human',sleep_=0.5):
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

        if self.state is not None:
            if self.viewer is None:
                self.screen_width, self.screen_height, self.pad = 800, 800, 0.1
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
                self.draw_scale = (self.screen_width - (self.pad * self.screen_width)) / self.dim \
                    if self.dim > self.dim else (self.screen_height - (
                        self.pad * self.screen_height)) / self.dim
                self.draw_start_x = self.screen_width / 2 - (self.draw_scale * self.dim) / 2
                self.draw_start_y = self.screen_height / 2 - (self.draw_scale * self.dim) / 2


                self.drawn_black, self.drawn_black_shift, self.recs = self.draw_blacks()
                                
                self.drawn_agents = self.draw_agents(fname='imgs/rocksample/rover.png')
                self.draw_grid()

            x,y = self.components['agents'][0].position
            self.drawn_agents[0].set_translation(self.draw_start_x+(x+self.drawn_black_shift)*self.draw_scale,self.draw_start_y+(y+self.drawn_black_shift)*self.draw_scale)


            self.draw_progress()
            self.viewer.render(return_rgb_array=mode == 'rgb_array')
            import time
            time.sleep(sleep_)

        return

    def draw_grid(self):
        grid = []
        linewidth = 2
        for x in range(self.dim + 1):
            grid.append(
                rendering.make_polyline([
                    (x * self.draw_scale + self.draw_start_x, self.draw_start_y),
                    (x * self.draw_scale + self.draw_start_x,
                     self.draw_start_y + self.dim * self.draw_scale)])
            )
            grid[-1].set_linewidth(linewidth)
            self.viewer.add_geom(grid[-1])

        for y in range(self.dim + 1):
            grid.append(
                rendering.make_polyline([
                    (self.draw_start_x, y * self.draw_scale + self.draw_start_y),
                    (self.draw_start_x + self.dim * self.draw_scale,
                     y * self.draw_scale + self.draw_start_y)])
            )
            grid[-1].set_linewidth(linewidth)
            self.viewer.add_geom(grid[-1])

        return grid

    def draw_blacks(self):
        drawn_blacks, shift, recs = [], 0, []
        shift = 0.5
        for cell in self.components['black']:
            drawn_blacks.append(rendering.Transform())
            x = 0.5*self.draw_scale
            rectangle = FilledPolygonCv4([(x,x),(-x,x),(-x,-x),(x,-x)])
            rectangle.add_attr(drawn_blacks[-1])
            (x,y) = cell
            drawn_blacks[-1].set_translation(self.draw_start_x+(x+shift)*self.draw_scale,self.draw_start_y+(y+shift)*self.draw_scale)

            self.viewer.add_geom(rectangle)

            recs.append(rectangle)

        return drawn_blacks, shift, recs

 
    def draw_progress(self):
        if self.episode != 0:
            self.viewer.geoms.pop()

        action = self.components['agents'][0].next_action
        obs = ""
        if self.episode > 0 and action == 4:
            obs = self.state['obs']

        state_ent = entropy(self.state['belief']/np.sum(self.state['belief']))
        progress_label = DrawText(pyglet.text.Label(
            'Episode: ' + str(self.episode) + ' | Action: ' + str(action) + ' | Observation : ' + str(obs) + ' | Entropy : '+ '%0.5f'%state_ent , \
            font_size=int(0.25 * self.draw_scale),
            x=self.draw_start_x, y=(1+self.dim) * self.draw_scale,
            anchor_x='left', anchor_y='center', color=(0, 0, 0, 255)))
        self.viewer.add_geom(progress_label)

    def draw_agents(self, type_='draw', fname=None):
        drawn_agent = []
        for agent in self.components['agents']:
            drawn_agent.append(rendering.Transform())
            try:
                with open(fname):
                    figure = rendering.Image(fname, \
                                                width=0.7 * self.draw_scale, height=0.7 * self.draw_scale)
                    figure.set_color(self.colors['lightgrey'][0], self.colors['lightgrey'][1],
                                            self.colors['lightgrey'][2])
                    figure.add_attr(drawn_agent[-1])
                    self.viewer.add_geom(figure)

                    x_shift,y_shift = 0.1 , 0.1

            except FileNotFoundError as e:
                raise e

        return drawn_agent

  