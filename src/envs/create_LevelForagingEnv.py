#####
# Imports
#####
from argparse import ArgumentParser, ArgumentTypeError
import itertools
import numpy as np
import random
from scenario_generator import get_env_types, get_initial_positions, save_LevelForagingEnv

#####
# Getting arguments
#####
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
parser.add_argument('--num_exp', dest='num_exp', default=0, type=int)
parser.add_argument('--dim', dest='dim', default=20, type=int)
parser.add_argument('--num_agents', dest='num_agents', default=7, type=int)
parser.add_argument('--num_tasks', dest='num_tasks', default=20, type=int)
#parser.add_argument('--po', dest='partial_observable', default=str2bool, type=bool)
#parser.add_argument('--display', dest='display', default=str2bool, type=bool)
args = parser.parse_args()

num_exp = args.num_exp
dim = args.dim
num_agents = args.num_agents
num_tasks = args.num_tasks

partial_observable = False
display = False

#####
# Creation Routine
#####
print('Creating Level Foraging Env #'+str(num_exp)+':', dim, num_agents, num_tasks)
# 1. Importing the environment and its necessary components
from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task

# 2. Defining the types and directions of the environment
types = get_env_types("LevelForagingEnv")
direction = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

# 3. Getting the initial positions
random_pos, num_tasks = get_initial_positions("LevelForagingEnv", dim, num_agents, num_tasks)

# 4. Creating the components
####
# SETTINGS INIT
####
agents, tasks = [], []

MIN_RADIUS, MAX_RADIUS = 0.5, 1.0
MIN_ANGLE, MAX_ANGLE = 0.5, 1.0 #(1.0/dim), 1.0

LEVELS, MIN_COMBINATION = [], None
MIN_LEVEL, MAX_LEVEL = 0.5, 1.0 
####
# END SETTINGS
####
# a. main agent
LEVELS.append(round(random.uniform(MIN_LEVEL, MAX_LEVEL), 3))
if not partial_observable:
    agents.append(
        Agent(index=str(0), atype='mcts',
                position=(random_pos[0][0], random_pos[0][1]),
                direction=random.sample(direction, 1)[0], 
                radius=MAX_RADIUS, angle=MAX_ANGLE,
                level=LEVELS[-1]))
else:
    agents.append(
        Agent(index=str(0), atype='pomcp',
                position=(random_pos[0][0], random_pos[0][1]),
                direction=random.sample(direction, 1)[0],
                radius=random.uniform(MIN_RADIUS, MAX_RADIUS),
                angle=random.uniform(MIN_ANGLE, MAX_ANGLE),
                level=LEVELS[-1]))
                
# b. teammates and tasks
for i in range(1, num_agents + num_tasks):
    if (i < num_agents):
        LEVELS.append(round(random.uniform(MIN_LEVEL, MAX_LEVEL), 3))
        agents.append( \
            Agent(index=str(i), atype=random.sample(types, 1)[0], 
                position=(random_pos[i][0], random_pos[i][1]),
                direction=random.sample(direction, 1)[0], 
                radius=random.uniform(MIN_RADIUS, MAX_RADIUS),
                angle=random.uniform(MIN_ANGLE, MAX_ANGLE),
                level=LEVELS[-1]))
    else:
        if MIN_COMBINATION is None:
            if num_agents > 4:
                MIN_COMBINATION = [list(c) for c in list(itertools.combinations(LEVELS, 4))]
                for j in range(len(MIN_COMBINATION)):
                    MIN_COMBINATION[j] = sum(MIN_COMBINATION[j])
                MIN_COMBINATION = min(MIN_COMBINATION) if min(MIN_COMBINATION) <= 1 else 1.0
            else:
                MIN_COMBINATION = sum(LEVELS) if sum(LEVELS) < 1 else 1.0

        sampleLevels = random.sample(LEVELS, 2)
        task_level = round(random.uniform(min(sampleLevels), MIN_COMBINATION), 3)
        tasks.append(Task(str(i), position=(random_pos[i][0], random_pos[i][1]), level=task_level))

# c. adding to components dict
components = {
    'agents': agents,
    'adhoc_agent_index': '0',
    'tasks': tasks}

# 5. Initialising the environment and returning it
if partial_observable:
    env = LevelForagingEnv((dim, dim), components, visibility='partial', display=display)
else:
    env = LevelForagingEnv((dim, dim), components, visibility='full', display=display)

save_LevelForagingEnv(env, dim, num_agents, num_tasks, num_exp)