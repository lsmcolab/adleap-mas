from copy import deepcopy
from gym import spaces
import numpy as np
import random as rd

from src.envs.AdhocReasoningEnv import AdhocAgent, AdhocReasoningEnv, StateSet


# Implementation based on "Heuristic Search Value Iteration
# for POMDP" : https://arxiv.org/pdf/1207.4166.pdf

class Agent(AdhocAgent):

    def __init__(self,index,position,type="random",radius=1):
        super(Agent,self).__init__(index,type)
        self.radius = radius # Do we need it ? 
        self.position = np.array(position)
        self.type = type

    def copy(self):
        copy_agent = Agent(self.index,self.position,self.type,self.radius)
        return copy_agent
    
    def set_parameter(self,parameters):
        raise NotImplementedError
    
    def get_parameter(self):
        raise NotImplementedError
        return
    

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

        obs = None
        if rd.random() < prob:
            if self.type=="good":
                obs = "good"
            else:
                obs = "bad"
        else:
            if self.type == "good":
                obs = "bad"
            else:
                obs = "good" 

        if obs=="good":
            self.belief = self.belief*prob/(self.belief*prob + (1-self.belief)*(1-prob))
        else:
            self.belief = self.belief*(1-prob)/(self.belief*(1-prob)+(1-self.belief)*prob)    
        return obs


def end_condition(state):
    return np.any(np.array(state['agent']) < 0)

def distance(pos_1,pos_2):
    return np.linalg.norm(pos_1-pos_2)

def update(env):
    env.state['agent'] = env.components['agents'][0].position
    env.state['rocks'] = []
    env.state['beliefs'] = []
    for rock in env.components['rocks']:
        env.state['rocks'].append(rock.position)
        env.state['beliefs'].append(rock.belief)
    
    return env.state

def do_action(env):
    env.state['obs'] = None
    action = env.components['agents'][0].next_action
    old_pos = env.components['agents'][0].position
    dim = env.dim
    x,y = old_pos[0], old_pos[1]
    info = {}
    new_pos = np.array([x,y])

    # EAST
    if action == 0:
        if x<dim-1:
            new_pos = np.array([x+1,y])
        else:
            new_pos = np.array([-1,-1])

    # WEST
    elif action == 1:
        new_pos = np.array([max(0,x-1),y])
    # NORTH
    elif action == 2:
        new_pos = np.array([x,min(y+1,dim-1)])
    
    # SOUTH
    elif action == 3:
        new_pos = np.array([x,max(0,y-1)])

    # SAMPLE
    elif action == 4:
        info['reward'] = -10
        for rock in env.components['rocks']:
            if (rock.position == new_pos).all():
                if rock.type == "good":
                    info['reward'] = 10
                    rock.type = "bad"
    else:
        rock_index = action-5
        rock = env.components['rocks'][rock_index]
        out = rock.check(new_pos)
        env.state['obs'] = (rock_index,out)

    env.components['agents'][0].position = new_pos

    next_state = update(env)
        
    return next_state,info

def rocksample_transition(action,real_env):
    real_env.components['agents'][0].next_action = action

    next_state,info = do_action(real_env)

    return next_state,info

def reward(state,next_state):
    if end_condition(next_state):
        return 10
    return 0

def environment_transformation(copied_env):
    for rock in copied_env.components['rocks']:
        rock.type = None
    copied_env.episode += 1
    return copied_env


class RockSampleState(spaces.Space):    
    def __init__(self,n):
        super(RockSampleState,self).__init__(shape=(n,n),dtype=np.float64)
    
    def sample(sef,seed=None):
        state = {'agent' : [], 'rocks':[], 'beliefs':[],'obs':None}
        return state

class RockSampleEnv(AdhocReasoningEnv):
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
        self.viewer = None
        self.display = display
        self.dim = dim
        state_set = StateSet(RockSampleState(dim),end_condition)
        transition_function = rocksample_transition
        self.no_of_rocks = len(components['rocks'])
        action_space = spaces.Discrete(4+1+self.no_of_rocks)
        reward_function = reward
        observation_space = environment_transformation

        # Initialising the Env
        self.state_dict = {
            0 : 'East',
            1 : 'West',
            2 : 'North',
            3 : 'South',
            4 : 'Sample'    
        }
        for i in range(5,5+self.no_of_rocks):
            self.state_dict[i] = "Sense-{}".format(i-5)

        super(RockSampleEnv,self).__init__(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)

        self.state_set.initial_components = self.copy_components(components)
        self.state_set.initial_state = {'agent':[],'rocks':[],'beliefs':[]}

        self.state_set.initial_state['agent'] = components['agents'][0].position
        
        for rock in components['rocks']:
            self.state_set.initial_state['rocks'].append(rock.position)
            self.state_set.initial_state['beliefs'].append(rock.belief)        
        
        self.state_set.initial_state['obs'] = None

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
        copied_env = RockSampleEnv(components, self.dim)
        copied_env.viewer = self.viewer
        copied_env.display = self.display
        copied_env.episode = self.episode

        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env
    
    # The environment is partially observable by definition
    def state_is_equal(self,state): 
        if not (self.state['agent']==state['agent']).all():
            return False
        
        if len(self.state['rocks'])!=len(state['rocks']):
            raise ValueError("Number of Rocks has changed")
        
        for pos1,pos2 in zip(self.state['rocks'],state['rocks']):
            if not (pos1==pos2).all():
                return False

        
        return (state['obs']==self.state['obs'])


    
    def observation_is_equal(self,obs):
       #        if np.linalg.norm(np.array(self.state['beliefs'])-np.array(state['beliefs'])) > 0.05:
#            return False
        state = obs.state
        return self.state_is_equal(state)

    def sample_state(self,agent,sample_p=0.1,n_sample=10):
        u_env = self.copy()
        obs = u_env.get_observation()

        for rock in obs.components['rocks']:
            rock.type = rd.sample(["good","bad"],1)[0]
        
        return obs
    
    def sample_nstate(self, agent, n, sample_a=0.1, sample_t=0.1, n_sample=10):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent,sample_a,sample_t,n_sample))
        return sampled_states

    def get_actions_list(self):
        return [i for i in range(0,self.no_of_rocks+5)]

    def get_adhoc_agent(self):
        return self.components['agents'][0]

    def render(self,mode="human",sleep_=0.5):
        #print("Agent Position : ", self.components['agents'][0].position)
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

                # Drawing the environment
                with open('imgs/rocksample/space.png'):
                    background = rendering.Image('imgs/rocksample/space.png', \
                                width=self.screen_width, height=self.screen_height)
                    background.add_attr(rendering.Transform(
                        translation=(int(self.screen_width)/2 + self.pad,
                                    int(self.screen_height)/2 + self.pad)))

                self.viewer.add_geom(background)

                self.drawn_rocks, self.drawn_rocks_shift, self.recs = self.draw_rocks(type_='figure', fname='imgs/rocksample/rock.jpeg')
                                
                self.drawn_agents = self.draw_agents(fname='imgs/rocksample/rover.png')
                self.draw_grid()

            x,y = self.components['agents'][0].position
            self.drawn_agents[0].set_translation(self.draw_start_x+(x+self.drawn_rocks_shift)*self.draw_scale,self.draw_start_y+(y+self.drawn_rocks_shift)*self.draw_scale)


            for i in range(len(self.components['rocks'])):
                x, y = self.components['rocks'][i].position[0], self.components['rocks'][i].position[1]
                self.drawn_rocks[i].set_translation(
                    self.draw_start_x + (x + self.drawn_rocks_shift) * self.draw_scale,
                    (y + self.drawn_rocks_shift) * self.draw_scale + self.draw_start_y
                )

            self.draw_progress()
            if self.episode > 0 and self.components['agents'][0].next_action > 4:
                rock_index = self.components['agents'][0].next_action-5
                start = (self.draw_start_x + (self.components['agents'][0].position[0]+self.drawn_rocks_shift)*self.draw_scale,\
                         self.draw_start_y + (self.components['agents'][0].position[1]+self.drawn_rocks_shift)*self.draw_scale)
                end = (self.draw_start_x + (self.components['rocks'][rock_index].position[0]+self.drawn_rocks_shift)*self.draw_scale,\
                         self.draw_start_y + (self.components['rocks'][rock_index].position[1]+self.drawn_rocks_shift)*self.draw_scale)
                
                if self.state['obs'] is None or self.state['obs'][1] not in ["good","bad"]:
                    print("Error : ",self.state['obs'],self.components['agents'][0].next_action)
                col = "red" if self.state['obs'][1] == "bad" else "green"
                self.viewer.draw_line(start,end,color=self.colors[col])
            
            if self.episode > 0 and self.components['agents'][0].next_action == 4:
                flag = None
                for idx,rock in enumerate(self.components['rocks']):
                    if (rock.position == self.components['agents'][0].position).all():
                        flag = True
                        break
                if flag:    
                    self.recs[idx].set_color(*self.colors['red'],1)

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

    def draw_rocks(self, type_='box', fname='imgs/rocksample/rock.jpeg'):
        drawn_rocks, shift, recs = [], 0, []
        shift = 0.5
        for rock in self.components['rocks']:
            drawn_rocks.append(rendering.Transform())
            try:
                with open(fname):
                    figure = rendering.Image(fname, \
                                                width=0.5 * self.draw_scale, height=0.5 * self.draw_scale)
                    figure.add_attr(drawn_rocks[-1])
                    x = 0.45*self.draw_scale
                    rectangle = FilledPolygonCv4([(x,x),(-x,x),(-x,-x),(x,-x)])
                    rectangle.add_attr(drawn_rocks[-1])
                    col = "red" if rock.type=="bad" else "green"
                    rectangle.set_color(self.colors[col][0],self.colors[col][1],self.colors[col][2],1)
                    self.viewer.add_geom(rectangle)

                    self.viewer.add_geom(figure)
                    recs.append(rectangle)
            except FileNotFoundError as e:
                raise e

        return drawn_rocks, shift, recs

 
    def draw_progress(self):
        if self.episode != 0:
            self.viewer.geoms.pop()

        action = self.components['agents'][0].next_action
        obs = ""
        if self.episode > 0 and action > 4:
            obs = self.state['obs']

        progress_label = DrawText(pyglet.text.Label(
            'Episode: ' + str(self.episode) + ' | Action: ' + str(action) + ' | Observation : ' + str(obs) , \
            font_size=int(0.25 * self.draw_scale),
            x=self.draw_start_x, y=(0.4) * self.draw_scale,
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

  