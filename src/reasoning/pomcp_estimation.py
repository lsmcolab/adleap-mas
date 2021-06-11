from math import *
from src.reasoning.fundamentals import Parameter
from numpy.random import choice
import random
import numpy as np
# import agent
import gc


radius_max = 1
radius_min = 0.1
angle_max = 1
angle_min = 0.1
level_max = 1
level_min = 0

types = ['l1', 'l2']

actions = ['L', 'N', 'E', 'S', 'W']
# totalItems=0
root = None


class PomcpConfig:
    def __init__(self, fundamental_values):

        self.fundamental_values = fundamental_values


class QTableRow:
    def __init__(self, action, q_value, sum_value, trials):
        self.action = action
        self.QValue = q_value
        self.sumValue = sum_value
        self.trials = trials

########################################################################################################################
class State:

    def __init__(self, simulator):
        self.simulator = simulator

    def equals(self, state):
        return self.simulator.equals(state.simulator)

################################################################################################################
class Node:

    def __init__(self, history, depth, state, belief_parameter=None, parent=None):

        self.parentNode = parent  # "None" for the root node
        self.history = history
        self.depth = depth
        self.childNodes = []
        self.cumulativeRewards = 0
        self.immediateReward = 0
        self.expectedReward = 0
        self.belief_parameter = belief_parameter
        self.state = state
        self.QTable = self.create_empty_table()
        self.visits = 0  # N(h)
        self.value = 0    # V(h)

    ####################################################################################################################
    @staticmethod
    def create_empty_table():
        Qt = list()
        Qt.append(QTableRow('L', 0.0, 0.0, 0))
        Qt.append(QTableRow('N', 0.0, 0.0, 0))
        Qt.append(QTableRow('E', 0.0, 0.0, 0))
        Qt.append(QTableRow('S', 0.0, 0.0, 0))
        Qt.append(QTableRow('W', 0.0, 0.0, 0))
        return Qt

    # ####################################################################################################################
    # def uct_select_child(self):
    #
    #     # UCB expects mean between 0 and 1
    #     s = sorted(self.childNodes, key=lambda c: c.expectedReward/self.numItems + sqrt(2 * log(self.visits) / c.visits))[-1]
    #     return s

    ###################################################################################################################

    def update(self, action, result):

        # TODO: We should change the table to a dictionary, so that we don't have to find the action
        for i in range(len(self.QTable)):
            if self.QTable[i].action == action:
                self.QTable[i].trials += 1
                self.QTable[i].sumValue += result
                #self.QTable[i].QValue = self.QTable[i].sumValue / self.QTable[i].trials
                self.QTable[i].QValue += (result - self.QTable[i].QValue) / self.QTable[i].trials
                return

####################################################################################################################

    def uct_select_action(self):

        maxUCB = -1
        maxA = None

        for a in range(len(self.QTable)):
            if self.valid(self.QTable[a].action):

                # TODO: The exploration constant could be set up in the configuration file
                if self.QTable[a].trials > 0:
                    current_ucb = self.QTable[a].QValue + 0.5 * sqrt(
                        log(float(self.visits)) / float(self.QTable[a].trials))

                else:
                    current_ucb = 0

                if current_ucb > maxUCB:
                    maxUCB = current_ucb
                    maxA = self.QTable[a].action

        if maxA is None:
            maxA = random.choice(actions)

        return maxA

#####################################################################################################################

    def valid(self, action):  # Check in order to avoid moving out of board.

        # if self.enemy:
        #     (x, y) = self.state.simulator.enemy_agent.get_position()
        # else:
        (x, y) = self.state.simulator.main_agent.get_position()

        m = self.state.simulator.dim_w
        n = self.state.simulator.dim_h
        for obstacle in self.state.simulator.obstacles:
            (x_o, y_o) = obstacle.get_position()
            if x == x_o and y == y_o:
                return False
        if x == 0:
            if action == 'W':
                return False

        if y == 0:
            if action == 'S':
                return False
        if x == m - 1:
            if action == 'E':
                return False

        if y == n - 1:
            if action == 'N':
                return False

        return True

#################################################################################################################
    def select_action(self):
        # If all *actions* of the current node have been tried at least once, then Select Child based on UCB


        # if self.untriedActions == []:
        return self.uct_select_action()
        #
        # # If there is some untried moves we will select a random move from the untried ones
        # if self.untriedActions!= []:
        #
        #     move = choice(self.untriedActions)
        #     self.untriedActions.remove(move)
        #     return move


################################################################################################################
class ONode(Node):

    def __init__(self,history, depth, state, belief_parameter,observation=None, parent=None):

        Node.__init__(self,history=history, depth=depth, state=state ,belief_parameter= belief_parameter, parent=parent)
        self.observation = observation  # the Action that got us to this node - "None" for the root node
        self.state = state
        self.particleFilter = []
        self.particleFilterCount = 100

        # self.numItems = state.simulator.items_left()
        # m = state.simulator.dim_w
        # n = state.simulator.dim_h
        adhoc_agent = state.simulator.get_adhoc_agent()
        (x, y) = adhoc_agent.position

        self.untriedActions = self.state.simulator.action_space.sample()

    def add_child(self, h, a):

        n = ANode(h, self.depth + 1, a, self)

        self.childNodes.append(n)

        return n

    ####################################################################################################################
    def initialise_data_set(self):
        # 1. Generating initial particle filters

        for i in range(0, self.particleFilterCount):
            # 2. Random uniform parameter sampling
            tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
            tmp_angle = random.uniform(angle_min, angle_max)  # 'angle'
            tmp_level = random.uniform(level_min, level_max)  # 'level'
            tmp_type = random.choice(types)
            pf = [tmp_level, tmp_radius, tmp_angle,  tmp_type]
            self.particleFilter.append(pf)

        return random.choice(self.particleFilter)

    ###################################################################################################################

    def best_parameter_type(self):
        parameters = []
        types = []
        # todo: should be implemented for all types
        l1_count = 0
        l2_count = 0
        for pf in self.particleFilter:
            parameters.append(pf[0:3])

            if pf[3] == 'l1':
                l1_count += 1
            if pf[3] == 'l2':
                l2_count += 1

            types.append(pf[3])

        if l2_count / len(self.particleFilter) > l1_count / len(self.particleFilter) :
            estimated_type = 'l2'
        else :
            estimated_type = 'l1'

        estimated_parameter = np.mean(np.array(parameters),axis=0)

        return estimated_parameter , estimated_type

    ####################################################################################################################
    def create_possible_actions(self,m,n,x, y):

        # if self.enemy:
        #     (x, y) = self.beliefState.simulator.enemy_agent.get_position()
        # else:
        untried_actions = ['N', 'S', 'E', 'W', 'L']

        if x == 0:
            untried_actions.remove('E')

        if y == 0:
            untried_actions.remove('S')

        if x == m - 1:
            untried_actions.remove('W')

        if y == m - 1:
            untried_actions.remove('N')

        return untried_actions


################################################################################################################
class ANode(Node):

    def __init__(self,history, depth,  action=None, parent=None):

        Node.__init__(self, history=history,depth=depth, state=None,  parent=parent)
        self.action = action  # the Action that got us to this node - "None" for the root node

    def add_child(self, h, s,o,p):

        n = ONode( history=h, state=s, observation=o, parent=self, depth=self.depth + 1,belief_parameter =p)
        # self.untriedActions.remove(a)
        self.childNodes.append(n)

        return n


########################################################################################################################
class POMCP:
    def __init__(self,  iteration_max, max_depth,particle_filter_numbers):

        self.iteration_max = iteration_max
        self.max_depth = max_depth
        self.particle_filter_numbers = particle_filter_numbers
        self.totalItems = 0


    ####################################################################################################################
    def do_move(self, sim, action, real=False):  # real: if it is the movement for real move ar the move for simulator

        # todo: elnaz: sim should change to the current state anr anything related to POMCP class
        tmp_m_agent = sim.main_agent

        get_reward = 0

        if action == 'L':
            load_item, (item_position_x, item_position_y) = tmp_m_agent.is_agent_face_to_item(sim)
            if load_item:
                destination_item_index = sim.find_item_by_location(item_position_x, item_position_y)

                if sim.items[destination_item_index].level <= tmp_m_agent.level:

                    sim.items[destination_item_index].loaded = True
                    get_reward += float(1.0)
                else:
                    sim.items[destination_item_index].agents_load_item.append(tmp_m_agent)
        else:
            (x_new, y_new) = tmp_m_agent.new_position_with_given_action(sim.dim_w, sim.dim_h, action)

            # If there new position is empty
            if sim.position_is_empty(x_new, y_new):
                tmp_m_agent.next_action = action
                tmp_m_agent.change_position_direction(sim.dim_w, sim.dim_h)
            else:
                tmp_m_agent.change_direction_with_action(action)

            sim.main_agent = tmp_m_agent

        sim.update_the_map()

        return get_reward

    ########################################################################################
    @staticmethod
    def terminal(state):
        # state_set.is_final_state(next_state)
        if state.simulator.state_set.is_final_state:
            return True
        return False

    ################################################################################################################
    def leaf(self,  node):

        if node.depth >= self.max_depth + 1:
            return True
        return False

    ################################################################################################################
    def simulate_action(self, state, action, belief_parameter):

        sim = state.simulator.copy()
        next_state = State(sim)

        # Run the A agent to get the actions probabilities

        for u_a in sim.agents:

            selected_type = belief_parameter[3]

            x, y = u_a.get_position()
            tmp_agent = agent.Agent(x, y, u_a.direction, selected_type, '-1')
            tmp_agent.set_parameters(sim, belief_parameter[0], belief_parameter[1], belief_parameter[2])

            sim.move_a_agent(tmp_agent)

        m_reward = self.do_move(sim, action)

        a_reward = sim.update_all_A_agents(sim)

        if sim.do_collaboration():
            c_reward = float(1)
        else:
            c_reward = 0

        total_reward = float(m_reward + a_reward + c_reward) / self.totalItems
        observation =next_state
        return next_state,observation, total_reward

    ################################################################################################################

    def find_new_root(self,previous_root, previous_action,previous_observation):

        if previous_root is None:
            return None

        root_node = None
        action_node = None
        # previous_action = current_state.simulator.main_agent.next_action

        for child in previous_root.childNodes:
            if child.action == previous_action:
                action_node = child
                break

        if action_node is None :
            return root_node

        for child in action_node.childNodes:
            if self.observation_is_equal(child.observation, previous_observation):
                root_node = child
                break

        return root_node

    ################################################################################################################
    def observation_is_equal(self, observation, other_observation):
        if len(observation) != len(other_observation):
            return False

        for o in range(len(observation)):
            if observation[o] not in other_observation:
                return False

        return True

    ################################################################################################################

    def rollout(self, state , history, depth,belief_parameter):

        if depth > self.max_depth or self.terminal(state):
            return 0

        # 1. Choosing the action
        # a ~ pi(h)

        action = random.choice(actions)

        # 2. Simulating the particle
        # (s',o,r) ~ G(s,a)
        (next_state, observation, reward) = self.simulate_action(state, action,belief_parameter)

        # 4. Calculating the reward
        R = reward + 0.95 * self.rollout(next_state,history, depth + 1,belief_parameter)

        return R

    ################################################################################################################

    def simulate(self, node):

        state = node.state

        belief_parameter = node.belief_parameter
        history = node.history

        if self.terminal(state):
            return 0
        if self.leaf(node):  # todo: why main_time_step
            return 0

        if node.childNodes == []:
            for action in ['L', 'N', 'E', 'S', 'W']:
                node.add_child(history,  action)

            # b. rollout
            return self.rollout(state, history, node.depth,belief_parameter)

        action = node.select_action()
        history.append(action)

        action_node = None

        for child in node.childNodes:
            if child.action == action:
                action_node = child
                break

        if action_node is None:
            action_node = node.add_child(history, action)

        (next_state, observation, reward) = self.simulate_action(state, action, belief_parameter)

        node.particleFilter.append(belief_parameter)
        history.append(observation)

        observation_node = None

        for child in action_node.childNodes:
            if child.state.equals(next_state):
                next_node = child
                break


        if observation_node is None:
            observation_node = action_node.add_child(history, next_state, observation,belief_parameter)

        discount_factor = 0.95
        q = reward + discount_factor * self.simulate(observation_node)

        node.update(action, q)
        node.visits += 1

        return q

    ####################################################################################################################
    def search(self, node):
        sim = node.state.simulator
        gc.collect()

        iteration_number = 0
        while iteration_number < self.iteration_max:
            if node.particleFilter == list():
                belief_parameter = node.initialise_data_set()
                node.history = []
            else:
                belief_parameter = node.particleFilter[random.randint(0, len(node.particleFilter) - 1)]

            # beliefState.simulator.draw_map()
            node.belief_parameter = belief_parameter

            # b. simulating
            self.simulate(node)

            iteration_number += 1
        return node.best_parameter_type()

    ####################################################################################################################

    def monte_carlo_planning(self, search_tree, simulator):
        global root
        current_state = State(simulator)
        self.adhoc_agent = current_state.simulator.get_adhoc_agent()
        previous_action = self.adhoc_agent.next_action

        previous_observation = current_state

        root_node = self.find_new_root(search_tree, previous_action, previous_observation)
        current_observation = current_state
        if root_node is None:
            root_node = ONode(None, depth=0, state=current_state, observation=current_observation,belief_parameter = None)

        node = root_node
        root = node

        estimated_parameter, estimated_type = self.search(root_node)  # main_time_step,

        return estimated_parameter, estimated_type

    ####################################################################################################################
    def start_estimation(self, search_tree, sim):

        for task in sim.components['tasks']:
            if task.completed:
                self.totalItems+=1

        tmp_sim = sim.copy()

        # if search_tree is None:
        #     totalItems = tmp_sim.items_left()

        estimated_parameter, estimated_type= self.monte_carlo_planning(search_tree, tmp_sim)
        new_estimated_parameter = Parameter(estimated_parameter.tolist()[0], estimated_parameter.tolist()[1],estimated_parameter.tolist()[2])

        return new_estimated_parameter, estimated_type
