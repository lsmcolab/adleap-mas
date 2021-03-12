from copy import deepcopy
from datetime import datetime
import gym
from gym import error, spaces
from gym.envs.classic_control import rendering
import numpy as np
import random as rd
import time

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

class DrawText(rendering.Geom):
    def __init__(self, label:pyglet.text.Label):
        rendering.Geom.__init__(self)
        self.label=label
    def render1(self):
        self.label.draw()

"""
    Ad-hoc 
"""
class Player(AdhocAgent):

    def __init__(self,index,atype):
        
        super(Player, self).__init__(index,atype)

        # agent parameters
        self.hand = []

    def copy(self):
        copy_agent = Player(self.index,self.type)
        for card in self.hand:
            copy_agent.hand.append(card)
        return copy_agent

"""
    Truco Environments 
"""
def end_condition(state):
    if state['round'] == [3,0]:
        if state['wins'].count(0) > state['wins'].count(1):
            state['points'][0] += 1
        else:
            state['points'][1] += 1
        return True 

    elif state['round'][0] == 2:
        if sum(state['wins'][0:2]) == 0:
            state['points'][0] += 1
            return True

        elif sum(state['wins'][0:2]) == 2:
            state['points'][1] += 1
            return True
        return False
        
    else:
        return False

def who_win(card1,card2,current_player,faceup_card):
    if  None in card1:
        return [card2[0],card2[1],current_player]
    if None in card2:
        return [card1[0],card1[1],current_player]

    if (card1[0] - 1) % 10 == faceup_card[0] and\
       (card2[0] - 1) % 10 != faceup_card[0]:
        return card1
    elif (card2[0] - 1) % 10 == faceup_card[0] and\
         (card1[0] - 1) % 10 != faceup_card[0]:
         return [card2[0],card2[1],current_player]
    elif (card1[0] - 1) % 10 == faceup_card[0] and\
         (card2[0] - 1) % 10 == faceup_card[0]:
        return card1 if card1[1] > card2[1] else [card2[0],card2[1],current_player]
    else:
        return card1 if card1[0] > card2[0] else [card2[0],card2[1],current_player]

def random_pick(action,hand):
    tmp_hand = deepcopy(hand)
    while None in tmp_hand[action]:
        rd.shuffle(tmp_hand)
    return hand.index(tmp_hand[action])

def truco_transition(action,real_env):
    if None in real_env.components['player'][real_env.current_player].hand[action]:
        info = {'invalid card reward':-1}
        action = real_env.action_space.sample()

    # playing the action
    real_env.state['winning card'] = \
        who_win(
            real_env.state['winning card'],
            real_env.components['player'][real_env.current_player].hand[action],
            real_env.current_player,
            real_env.state['face up card'])

    # returning the next state and information
    info = {'player':real_env.current_player,'card':action}
    real_env.components['player'][real_env.current_player].hand[action] = [None,None]
    next_state = real_env.state

    # updating the round
    if real_env.state['round'][1] == 3:
        real_env.state['wins'][real_env.state['round'][0]] = real_env.state['winning card'][2] % 2
        real_env.state['round'][0] = (real_env.state['round'][0] + 1)
        real_env.state['round'][1]  = 0

        real_env.current_player = real_env.state['winning card'][2]
        real_env.state['winning card'] = [None,None,None]
    else:
        real_env.state['round'][1] += 1
        real_env.current_player = (real_env.current_player + 1) % 4

    return next_state, info

def reward(state,next_state):
    if (next_state['winning card'][2] == state['winning card'][2]) or\
        (None in state['winning card']) or (None in next_state['winning card']):
        return 0
    elif next_state['winning card'][0] == next_state['face up card'][0] + 1:
        if ((next_state['winning card'][0] - state['winning card'][0]) +\
        (next_state['winning card'][1] - state['winning card'][1])) > 0:
            return 1
        else:
            return 0
    else:
        if (next_state['winning card'][0] - state['winning card'][0]) > 0:
            return 1
        else:
            return 0 

def truco_observation(copied_env):
    # reseting the cards in game
    copied_env.components['cards in game'] = [[x,y] \
        for x in range(len(copied_env.cards)) for y in range(len(copied_env.suits))]
    rd.shuffle(copied_env.components['cards in game'])
    
    # removing visible cards (own and faced up cards)
    for card in copied_env.components['player'][copied_env.current_player].hand:
        if card in copied_env.components['cards in game']:
            copied_env.components['cards in game'].remove(card)
    copied_env.components['cards in game'].remove(copied_env.state['face up card'])

    # blinding the others players' hand if the game is partial observable
    if copied_env.visibility == 'partial':
        for player in copied_env.components['player']:
            if player != copied_env.components['player'][copied_env.current_player]:
                player.hand = [[None,None],[None,None],[None,None]]

    return copied_env

class TrucoEnv(AdhocReasoningEnv):

    action_dict = {0:'Card1',1:'Card2',2:'Card3'}

    cards = {0:'4',1:'5',2:'6',3:'7',4:'Q',5:'J',6:'K',7:'A',8:'2',9:'3'}
    suits = {0:'D',1:'S',2:'H',3:'C'}

    total_cards = len(cards)*len(suits)

    def __init__(self,players,reasoning,visibility='full'): 
        self.viewer = None
        self.round_cards = []
        self.visibility = visibility

        # Defining the Ad-hoc Teamwork Env parameters
        state_set = StateSet(
            spaces.Dict({
                'face up card':spaces.MultiDiscrete([len(self.cards),len(self.suits)]),
                'winning card':spaces.MultiDiscrete([len(self.cards),len(self.suits),len(players)]),
                'round':spaces.MultiDiscrete([3,4]),
                'wins':spaces.MultiBinary(3),
                'points':spaces.MultiDiscrete([12,12])
            }),
            end_condition)
        transition_function = truco_transition
        action_space = spaces.Discrete(3)
        action_space.sample = self.random_pick

        reward_function = reward
        observation_space = truco_observation
        components = {'player':[
                        Player(players[0],reasoning[0]),
                        Player(players[1],reasoning[1]),
                        Player(players[2],reasoning[2]),
                        Player(players[3],reasoning[3])],
                     'cards in game':[(x,y) for x in range(len(self.cards)) for y in range(len(self.suits))],
                    }
        
        super(TrucoEnv, self).__init__(state_set,\
         transition_function, action_space, reward_function,\
                                observation_space, components)

        # Setting the inital state
        self.n_players = len(players)
        self.start_player = 0
        self.current_player = 0
        self.state_set.initial_state = {
            'face up card':[None,None],
            'winning card':[None,None,None],
            'round':[0,0],
            'wins':[0,0,0],
            'points':[0,0]
        }

    def reset(self):
        # Reset the state of the environment to an initial state
        self.episode = 0
        if self.state_set.initial_state is not None:
            self.state = deepcopy(self.state_set.initial_state)
            self.deal()
            return self.observation_space(self.copy())
        else:
            raise ValueError("the initial state from the state set is None.")

    def copy(self):
        copied_env = TrucoEnv(players = [player.index for player in self.components['player']],
                reasoning = [player.type for player in self.components['player']], visibility=self.visibility)
        copied_env.start_player = self.start_player
        copied_env.current_player = self.current_player
        copied_env.state = deepcopy(self.state_set.initial_state)
        
        copied_env.state['face up card'] = [None,None] if None in self.state['face up card'] \
                                            else [info for info in self.state['face up card']]
        copied_env.state['winning card'] = [None,None,None] if None in self.state['winning card'] \
                                            else [info for info in self.state['winning card']]
        copied_env.state['round'] = [info for info in self.state['round']]
        copied_env.state['wins'] = [info for info in self.state['wins']]
        copied_env.state['points'] = [info for info in self.state['points']]

        for i in range(len(copied_env.components['player'])):
            player = (copied_env.components['player'][i])
            player.hand = [card for card in self.components['player'][i].hand]
            for card in player.hand:
                if card in copied_env.components['cards in game']:
                    copied_env.components['cards in game'].remove(tuple(card))
        
        copied_env.components['cards in game'].remove(tuple(copied_env.state['face up card']))

        return copied_env

    def random_pick(self):
        action, hand = \
            0, self.components['player'][self.current_player].hand

        tmp_hand = deepcopy(hand)

        if all([list(card) == [None,None] for card in tmp_hand]):
            hand[0] = self.components['cards in game'].pop()
            tmp_hand[0] = hand[0]

        while None in tmp_hand[action]:
            rd.shuffle(tmp_hand)

        return hand.index(tmp_hand[action])

    def state_is_equal(self,state):
        if self.state['face up card'][0] == state['face up card'][0] and \
           self.state['face up card'][1] == state['face up card'][1] and \
           self.state['winning card'][0] == state['winning card'][0] and \
           self.state['winning card'][1] == state['winning card'][1] and \
           self.state['winning card'][2] == state['winning card'][2] and \
           self.state['round'][0] == state['round'][0] and\
           self.state['round'][1] == state['round'][1] and\
           self.state['wins'][0] == state['wins'][0] and \
           self.state['wins'][1] == state['wins'][1] and \
           self.state['wins'][2] == state['wins'][2] and \
           self.state['points'][0] == state['points'][0] and \
           self.state['points'][1] == state['points'][1]:
            return True

    def get_adhoc_agent(self):
        return self.components['player'][self.current_player]

    def deal(self):
        self.state['face up card'] = [None,None]
        self.state['winning card'] = [None,None,None]
        self.state['round'] = [0,0]
        self.state['wins'] = [0,0,0]
        
        self.components['cards in game'] = [[x,y] for x in range(len(self.cards)) for y in range(len(self.suits))]
        rd.shuffle(self.components['cards in game'])

        self.state['face up card'] = self.components['cards in game'].pop(0)

        for i in range(self.n_players):
            self.components['player'][i].hand = []
            while len(self.components['player'][i].hand) < 3:
                self.components['player'][i].hand.append(self.components['cards in game'].pop(0))

    def observation_is_equal(self,obs):
        observable_env = self.observation_space(self.copy())
        if observable_env.state['winning card'][0] == obs.state['winning card'][0] and\
            observable_env.state['winning card'][1] == obs.state['winning card'][1] and\
            observable_env.state['winning card'][1] == obs.state['winning card'][1]:
                return True
        return False

    def sample_state(self,player):
        # 1. Defining the base simulation
        u_env = self.copy()

        # 2. Sampling random cards for the players
        for player in u_env.components['player']:
            for i in range(len(player.hand)):
                player.hand[i] = u_env.components['cards in game'].pop(0)

        return u_env

    def render(self, mode='human', info={}, agents_color={}):
        # Render the environment to the screen
        self.agents_color = agents_color
        if self.state is not None:
            if self.viewer is None:
                self.screen_width, self.screen_height = 1000, 1000
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

                # table background
                self.table_background = rendering.FilledPolygon([(0,0),(0,self.screen_height),
                                (self.screen_width,self.screen_height),(self.screen_width,0)])
                self.table_background.set_color(r=0.12, g=0.6, b=0.1) # green
                self.viewer.add_geom(self.table_background)

                # table circle
                self.table_circle = rendering.make_circle(self.screen_width/4,res=50,filled=False)
                self.table_circle.set_color(r=0.6, g=0.0, b=0.0) # red
                self.table_circle.add_attr(rendering.Transform(translation=(self.screen_width/2,self.screen_height/2)))
                self.table_circle.set_linewidth(5)
                self.viewer.add_geom(self.table_circle)

        if self.state['round'] == [0,0]:
            self.round_cards = []
            self.viewer.geoms = self.viewer.geoms[0:2]

            # points 
            self.points_bg = rendering.FilledPolygon(
                [(0,0),(0,0.1*self.screen_height),(0.2*self.screen_width,0.1*self.screen_height),(0.2*self.screen_width,0)])
            self.points_bg.add_attr( rendering.Transform(translation=(0.11*self.screen_width, 0.825*self.screen_height)))
            self.points_bg.set_color(r=1, g=1, b=1) # white
            self.viewer.add_geom( self.points_bg)

            for i in range(3):
                round_points = rendering.make_circle(0.1*self.screen_width/6)
                round_points.set_color(r=0.5, g=0.5, b=0.5) # grey
                round_points.add_attr( rendering.Transform(translation=(0.15*self.screen_width + i*(0.3*self.screen_width/6) , 0.85*self.screen_height)))
                self.viewer.add_geom(round_points)

            self.points_label = DrawText(pyglet.text.Label('Scoreboard: '+str(self.state['points'][0])+'x'+str(self.state['points'][1]),
                font_size=int(0.1*self.screen_height/6),x=0.2*self.screen_width, y=0.9*self.screen_height,\
                     anchor_x='center', anchor_y='center', color=(0, 0, 0, 255)))
            self.viewer.add_geom( self.points_label)

            self.points_label_redid = rendering.PolyLine([(0.25*self.screen_width, 0.89*self.screen_height),(0.27*self.screen_width, 0.89*self.screen_height)],False)
            self.points_label_redid.set_linewidth(5)
            self.points_label_redid.set_color(r=0.6, g=0.0, b=0.0) # red
            self.viewer.add_geom( self.points_label_redid)

            self.points_label_blueid = rendering.PolyLine([(0.27*self.screen_width, 0.89*self.screen_height),(0.29*self.screen_width, 0.89*self.screen_height)],False)
            self.points_label_blueid.set_linewidth(5)
            self.points_label_blueid.set_color(r=0.0, g=0.0, b=0.6) # blue
            self.viewer.add_geom( self.points_label_blueid)

            
            # deal - players and cards
            self.drawn_players, self.drawn_cards = self.draw_players()
            self.viewer.render(return_rgb_array=mode == 'rgb_array')
            
            for i in range(len(self.components['player'])):
                screen_position = { 0:(self.screen_width/2, 0.1*self.screen_height),
                                1:(0.9*self.screen_width, self.screen_height/2),
                                2:(self.screen_width/2, 0.9*self.screen_height),
                                3:(0.1*self.screen_width, self.screen_height/2)}
                rotate = {  0:0,
                            1:np.pi/2,
                            2:np.pi,
                            3:3*np.pi/2}
                x, y = screen_position[i]

                for j in range(3):
                    index = (3*i) + j + 1
                    self.drawn_cards[index].add_attr( rendering.Transform( rotation=(rotate[i]) ) )
                    self.drawn_cards[index].add_attr( rendering.Transform( translation=(x,y) ) )
                    self.viewer.render(return_rgb_array=mode == 'rgb_array')
            
        else:
            i = info['player']
            j = info['card']
            index = (3*i) + j + 1

            if self.state['round'][0] > 0 and self.state['round'][1] == 1:
                for card_index in self.round_cards:
                    try:
                        fname = 'imgs/cards/red_back.png'
                        with open(fname):
                            figure = rendering.Image(fname,\
                                width=0.9*self.screen_width/11,height=0.9*self.screen_height/8)

                            for attr in self.drawn_cards[card_index].__getattribute__('attrs'):
                                figure.add_attr(attr)

                            self.drawn_cards[card_index] = figure
                            self.viewer.geoms[card_index-13] = figure

                    except FileNotFoundError as e:
                        raise e

                if self.state['wins'][self.state['round'][0]-1] == 0:
                    self.viewer.geoms[2+self.state['round'][0]].set_color(r=0.6, g=0.0, b=0.0) # red
                else:
                    self.viewer.geoms[2+self.state['round'][0]].set_color(r=0.0, g=0.0, b=0.6) # blue

                self.round_cards = []
                self.viewer.render(return_rgb_array=mode == 'rgb_array')
                time.sleep(1)
            
            self.round_cards.append(index)
            move_x = 200 if i == 3 else 0
            move_x -= 200 if i == 1 else 0
            move_y = 200 if i == 0 else 0
            move_y -= 200 if i == 2 else 0

            self.drawn_cards[index].add_attr( rendering.Transform( translation=(move_x,move_y) ) )
            self.viewer.render(return_rgb_array=mode == 'rgb_array')

        time.sleep(2)
        return 

    def draw_players(self):
        drawn_player = []
        drawn_cards = []

        # players name
        for i in range(len(self.components['player'])):
            drawn_player.append(rendering.Transform())

            player_colors = rendering.make_circle(0.1*self.screen_width/6)
            if i % 2 == 0:
                player_colors.set_color(r=0.6, g=0.0, b=0.0) # red
            else:
                player_colors.set_color(r=0.0, g=0.0, b=0.6) # blue
            player_colors.add_attr( rendering.Transform(translation=(0,100)) )
            player_colors.add_attr(drawn_player[i])
            self.viewer.add_geom(player_colors)

            label = DrawText(pyglet.text.Label(str(self.components['player'][i].index),
                font_size=int(0.1*self.screen_height/6),x=0, y=100, anchor_x='center', anchor_y='center', color=(255, 255, 255, 255)))
            label.add_attr(drawn_player[i])
            self.viewer.add_geom(label)

            screen_position = { 0:(self.screen_width/2, 0.1*self.screen_height),
                                1:(0.9*self.screen_width, self.screen_height/2),
                                2:(self.screen_width/2, 0.9*self.screen_height),
                                3:(0.1*self.screen_width, self.screen_height/2)}
            rotate = {  0:0,
                        1:np.pi/2,
                        2:np.pi,
                        3:3*np.pi/2}
            x, y = screen_position[i]
            drawn_player[i].set_rotation(rotate[i])
            drawn_player[i].set_translation(x,y)

        # cards
        try:
            fname = 'imgs/cards/'+str(self.cards[self.state['face up card'][0]])+str(self.suits[self.state['face up card'][1]])+'.png'
            with open(fname):
                figure = rendering.Image(fname,\
                    width=0.9*self.screen_width/11,height=0.9*self.screen_height/8)
                figure.add_attr(
                    rendering.Transform(translation=((self.screen_width/2,self.screen_height/2)))
                )
                drawn_cards.append(figure)
                self.viewer.add_geom(figure)

        except FileNotFoundError as e:
            raise e

        for i in range(len(self.components['player'])):
            for j in range(len(self.components['player'][i].hand)):
                try:
                    fname = 'imgs/cards/'+str(self.cards[self.components['player'][i].hand[j][0]])+str(self.suits[self.components['player'][i].hand[j][1]])+'.png'
                    with open(fname):
                        figure = rendering.Image(fname,\
                            width=0.9*self.screen_width/11,height=0.9*self.screen_height/8)
                        figure.add_attr(
                            rendering.Transform(translation=((j*0.9*self.screen_width/11) - (len(self.components['player'][i].hand)*0.9*self.screen_width/22),0))
                        )
                        drawn_cards.append(figure)
                        self.viewer.add_geom(figure)
        
                except FileNotFoundError as e:
                    raise e

        return drawn_player, drawn_cards

    def feature(self):
        actual_state = self.state
        a = []
        for key in actual_state.keys():
            for val in actual_state[key]:
                if (val is None):
                    a.append(-1)
                else:
                    a.append(val)
        for val in self.components['player'][self.current_player].hand:
            if None in val:
                a.append(-1)
                a.append(-1)
            else:
                a.append(val[0])
                a.append(val[1])

        return np.array(a)