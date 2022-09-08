from copy import deepcopy
from gym import spaces
import numpy as np
import os
import random as rd

from .AdhocReasoningEnv import AdhocReasoningEnv, AdhocAgent, StateSet

"""
    Support classes
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
    return False

def who_win(card1,card2,faceup_card):
    if  None in card1:
        return [card2[0],card2[1]]
    if None in card2:
        return [card1[0],card1[1]]

    if (card1[0] - 1) % 10 == faceup_card[0] and\
       (card2[0] - 1) % 10 != faceup_card[0]:
        return card1
    elif (card2[0] - 1) % 10 == faceup_card[0] and\
         (card1[0] - 1) % 10 != faceup_card[0]:
         return [card2[0],card2[1]]
    elif (card1[0] - 1) % 10 == faceup_card[0] and\
         (card2[0] - 1) % 10 == faceup_card[0]:
        return card1 if card1[1] > card2[1] else [card2[0],card2[1]]
    else:
        return card1 if card1[0] > card2[0] else [card2[0],card2[1]]

def finish_round(env):
    # defining the winning card
    winning_card = [None,None]
    for i in range(len(env.state['played cards'])):
        winning_card = who_win(env.state['played cards'][i],winning_card,env.state['face up card'])
            
    # defining the next player
    next_player = None
    for i in range(len(env.state['played cards'])):
        if winning_card[0] == env.state['played cards'][i][0] and\
         winning_card[1] == env.state['played cards'][i][1]:
            next_player = i
    env.current_player = next_player

    # reseting played cards
    env.state['played cards'] = []

    # checking the best of three and counting the round
    env.round_points[env.round] = (env.current_player % 2)
    env.round += 1
    
    teamA_rp = env.round_points.count(0)
    teamB_rp = env.round_points.count(1)
    if teamA_rp == 2:
        env.points[0] += 1
        env.round = 0
        env.start_player = (env.start_player + 1) % (len(env.components['players']))
        env.current_player = env.start_player
        env.empty_hands()
    elif teamB_rp == 2:
        env.points[1] += 1
        env.round = 0
        env.start_player = (env.start_player + 1) % (len(env.components['players']))
        env.current_player = env.start_player
        env.empty_hands()

    return env

def truco_transition(action,real_env):
    info = None

    if real_env.round == 0 and real_env.hands_are_empty():
        print('Dealing')
        real_env.deal()
    else:
        print('Playing')
        agent = real_env.get_adhoc_agent()
        print(agent.hand)
        card = agent.hand[action]
        agent.hand[action] = None
        real_env.state['played cards'].append(card)

    if len(real_env.state['played cards']) == 4:
        real_env = finish_round(real_env)
    else:
        real_env.current_player = (real_env.current_player + 1) % (len(real_env.components['players']))
    return real_env, info

def reward(state,next_state):
    return 0 

def environment_transformation(copied_env):
    return copied_env

class TrucoState(spaces.Space):    
    def __init__(self):
        super(TrucoState,self).__init__(dtype=list)
    
    def sample(sef,seed=None):
        raise NotImplemented 

class TrucoEnv(AdhocReasoningEnv):

    action_dict = {
        0:'PlayCard1',
        1:'PlayCard2',
        2:'PlayCard3'
    }

    cards = {0:'4',1:'5',2:'6',3:'7',4:'Q',5:'J',6:'K',7:'A',8:'2',9:'3'}
    suits = {0:'D',1:'S',2:'H',3:'C'}

    total_cards = len(cards)*len(suits)

    def __init__(self,components,display=False):
        ###
        # Env Settings
        ###
        self.n_players = len(components['players'])

        self.round = 0
        self.points = [0,0]
        self.round_points = [None,None,None]
        self.start_player = 0
        self.current_player = 0

        self.deck = [[x,y] for x in range(len(self.cards)) for y in range(len(self.suits))]
        rd.shuffle(self.deck)
        self.played_cards = []

        state_set = StateSet(TrucoState(),end_condition)
        transition_function = truco_transition
        action_space = spaces.Discrete(3)
        reward_function = reward
        observation_space = environment_transformation

        ###
        # Initialising the env
        ###
        super(TrucoEnv,self).__init__(state_set,\
                                transition_function,action_space,reward_function,\
                                    observation_space,components)
        self.state_set.initial_components = self.copy_components(components)
        self.state_set.initial_state = {
                'face up card':[None,None],
                'played cards':[]
            }
            
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

    def get_trans_p(self,action):
        return [self,1]

    def get_obs_p(self,action):
        return [self,1]

    def import_method(self, agent_type):
        from importlib import import_module
        try:
            module = import_module('src.reasoning.truco.'+agent_type)
        except:
            module = import_module('src.reasoning.'+agent_type)

        method = getattr(module, agent_type+'_planning')
        return method

    def copy(self):
        components = self.copy_components(self.components)
        copied_env = TrucoEnv(components,self.display)
        copied_env.screen = self.screen
        copied_env.episode = self.episode
        copied_env.renderer = self.renderer
        # Setting the initial state
        copied_env.state_set.initial_state = self.copy_components(self.state_set.initial_state)
        copied_env.state = deepcopy(self.state)
        copied_env.simulation = self.simulation
        return copied_env

    def hands_are_empty(self):
        for i in range(len(self.components['players'])):
            for card in self.components['players'][i].hand:
                if card is not None:
                    return False
        return True

    def empty_hands(self):
        for i in range(len(self.components['players'])):
            self.components['players'][i].hand = [None,None,None]

    def deal(self):
        # reseting the table
        self.state['face up card'] = [None,None]
        self.state['played cards'] = []
                
        self.round = 0
        self.round_points = [None,None,None]
        
        # shuffling the deck
        self.deck = [[x,y] for x in range(len(self.cards)) for y in range(len(self.suits))]
        rd.shuffle(self.deck)
        self.played_cards = []

        # playing the manilha
        self.state['face up card'] = self.deck.pop(0)

        # dealing players cards
        for i in range(self.n_players):
            self.components['players'][i].hand = []
            while len(self.components['players'][i].hand) < 3:
                self.components['players'][i].hand.append(self.deck.pop(0))
    
    # The environment is partially observable by definition
    def state_is_equal(self,state):    
        if self.state['face up card'][0] != state['face up card'][0] or \
           self.state['face up card'][1] == state['face up card'][1]:
            return False   

        for card in self.state['played cards']:
            if card not in state['played cards']:
                return False

        return True

    def observation_is_equal(self,obs):
        return self.state_is_equal(obs)

    def sample_state(self,agent):
        # 1. Defining the base simulation
        u_env = self.copy()
        u_env.deck = [[x,y] for x in range(len(self.cards)) for y in range(len(self.suits))]
        rd.shuffle(u_env.deck)

        # 2. Sampling random cards for the players
        for player in u_env.components['player']:
            if player.index != agent.index:
                for i in range(len(player.hand)):
                    player.hand[i] = u_env.deck.pop(0)

        return u_env
    
    def sample_nstate(self, agent, n):
        sampled_states = []
        while len(sampled_states) < n:
            sampled_states.append(self.sample_state(agent))
        return sampled_states

    def get_actions_list(self):
        return [i for i in range(len(self.action_dict))]

    def get_adhoc_agent(self):
        return self.components['players'][self.current_player]

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

        # table
        game_width, game_height = 700, 700
        self.game_surf = pygame.Surface((game_width, game_height))
        self.game_surf.fill(self.colors['darkgreen'])
       
        inner_circle_r = 0.2
        while inner_circle_r < 0.21:
            gfxdraw.aacircle(self.game_surf,int(0.5*game_width),int(0.5*game_height),
                int((inner_circle_r)*np.sqrt(game_width**2 + game_height**2)),self.colors['red'])
            inner_circle_r += 0.001

        # face up card
        if self.state['face up card'][0] is not None and\
         self.state['face up card'][1] is not None:
            card = self.cards[self.state['face up card'][0]]
            suit = self.suits[self.state['face up card'][1]]
            card_ret = pygame.Rect((0.45*game_width, 0.6*game_height),\
                            (int(0.1*game_width),int(0.15*game_height)))
            card_img = pygame.image.load(os.path.abspath("./imgs/cards/"+card+suit+".png"))
            card_img = pygame.transform.flip(card_img,False,True)
            card_img = pygame.transform.scale(card_img, card_ret.size)
            card_img = card_img.convert_alpha()
            card_img.fill((255, 255, 255, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self.game_surf.blit(card_img,card_ret)

        # players
        players_screen_pos = [\
            [0.5*game_width,0.1*game_height],
            [0.9*game_width,0.5*game_height],
            [0.5*game_width,0.9*game_height],
            [0.1*game_width,0.5*game_height]
            ]
        players_rotate = [180,0,0,0]
        myfont = pygame.font.SysFont("Ariel", 35)
        for i in range(len(self.components['players'])): 
            # name
            color = 'red' if (i % 2) == 0 else 'blue'
            label = myfont.render(self.components['players'][i].index, True, self.colors[color])
            label = pygame.transform.rotate(label,players_rotate[i])
            self.game_surf.blit(label, 
                (players_screen_pos[i][0]-myfont.size(self.components['players'][i].index)[0]/2,
                 players_screen_pos[i][1]-myfont.size(self.components['players'][i].index)[1]/2))
            
            # hand
            for j in range(len(self.components['players'][i].hand)):
                if self.components['players'][i].hand[j] is not None:
                    card = self.cards[self.components['players'][i].hand[j][0]]
                    suit = self.suits[self.components['players'][i].hand[j][1]]
                    card_ret = pygame.Rect((
                        players_screen_pos[i][0]-(0.1*game_width)+(0.05*j*game_width),
                        players_screen_pos[i][1]-(0.05*game_height)),\
                                                        (int(0.1*game_width),int(0.15*game_height)))
                    card_img = pygame.image.load(os.path.abspath("./imgs/cards/"+card+suit+".png"))
                    card_img = pygame.transform.flip(card_img,False,True)
                    card_img = pygame.transform.scale(card_img, card_ret.size)
                    card_img = card_img.convert_alpha()
                    card_img.fill((255, 255, 255, 150), special_flags=pygame.BLEND_RGBA_MULT)
                    self.game_surf.blit(card_img,card_ret)
        
        # played cards
        for i in range(len(self.state['played cards'])):
            card = self.cards[self.state['played cards'][i][0]]
            suit = self.suits[self.state['played cards'][i][1]]
            card_ret = pygame.Rect((0.4*game_width+(0.05*i*game_width), 0.4*game_height),\
                            (int(0.1*game_width),int(0.15*game_height)))
            card_img = pygame.image.load(os.path.abspath("./imgs/cards/"+card+suit+".png"))
            card_img = pygame.transform.flip(card_img,False,True)
            card_img = pygame.transform.scale(card_img, card_ret.size)
            card_img = card_img.convert_alpha()
            card_img.fill((255, 255, 255, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self.game_surf.blit(card_img,card_ret)
        
        # points
        myfont = pygame.font.SysFont("Ariel", 35)
        labelA = myfont.render("[Team A] %02d x" % (self.points[0]), True, self.colors['red'])
        labelB = myfont.render("x %02d [Team B]" % (self.points[1]), True, self.colors['blue'])
        self.screen.blit(labelA, (10, 10))
        self.screen.blit(labelB, (150, 10))

        for i in range(len(self.round_points)):
            if self.round_points[i] is None:
                gfxdraw.filled_circle(self.screen,400+40*i,20,10,self.colors['darkgrey'])
            elif self.round_points[i] == 0:
                gfxdraw.filled_circle(self.screen,400+40*i,20,10,self.colors['red'])
            else:
                gfxdraw.filled_circle(self.screen,400+40*i,20,10,self.colors['blue'])

        ##
        # Displaying
        ##
        self.game_surf = pygame.transform.flip(self.game_surf, False, True)
        self.screen.blit(self.game_surf, (50, 50))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )