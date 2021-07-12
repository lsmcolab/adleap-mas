import itertools
import numpy as np
import pickle
import random


def get_initial_positions(env_name, dim, nagents, ntasks):
    # verifying if it is possible to create the scenario
    if env_name == "LevelForagingEnv" and (ntasks * 9) > (dim ** 2):
        raise StopIteration

    # getting the random positions
    pos = []
    while len(pos) < (nagents + ntasks):
        x = random.randint(0, dim - 1)
        y = random.randint(0, dim - 1)

        if env_name == "LevelForagingEnv":
            if len(pos) >= nagents:
                if x > 0 and x < dim and y > 0 and y < dim and \
                        (x, y) not in pos and (x + 1, y) not in pos and \
                        (x + 1, y + 1) not in pos and (x, y + 1) not in pos and \
                        (x - 1, y + 1) not in pos and (x - 1, y) not in pos and \
                        (x - 1, y - 1) not in pos and (x, y - 1) not in pos and \
                        (x + 1, y - 1) not in pos:
                    pos.append((x, y))
            else:
                if (x, y) not in pos:
                    pos.append((x, y))
        elif env_name == "CaptureEnv":
            if (x, y) not in pos:
                pos.append((x, y))
        else:
            raise NotImplemented

    return pos


def get_env_types(env_name):
    if env_name == "LevelForagingEnv":
        return ['l1', 'l2'] # , 'l3'
    elif env_name == "CaptureEnv":
        return ['c1', 'c2'] # , 'c3'
    else:
        raise NotImplemented


def get_env_nparameters(env_name):
    if env_name == "LevelForagingEnv":
        return 3
    elif env_name == "CaptureEnv":
        return 2
    else:
        raise NotImplemented


def save_LevelForagingEnv(env, dim, num_agents, num_tasks, num_exp):
    file = open(
        './src/envs/maps/LevelForagingEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(num_exp) + '.pickle',
        'wb')
    env = pickle.dump(env, file)
    file.close()
    return env


def load_LevelForagingEnv(dim, num_agents, num_tasks, num_exp):
    with open('./src/envs/maps/LevelForagingEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(
            num_exp) + '.pickle', 'rb') as map:
        env = pickle.load(map)
    return env


def create_LevelForagingEnv(dim, num_agents, num_tasks, partial_observable=False, display=False, num_exp=0):
    # 1. Importing the environment and its necessary components
    from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task

    # 2. Defining the types and directions of the environment
    types = get_env_types("LevelForagingEnv")
    direction = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # 3. Getting the initial positions
    random_pos = get_initial_positions("LevelForagingEnv", dim, num_agents, num_tasks)

    # 4. Creating the components
    agents, tasks = [], []

    # a. main agent
    if not partial_observable:
        agents.append(
            Agent(index=str(0), atype='mcts',
                  position=(random_pos[0][0], random_pos[0][1]),
                  direction=random.sample(direction, 1)[0], radius=1.0, angle=1.0,
                  level=round(random.uniform(0.1, 1), 3)))
    else:
        agents.append(
            Agent(index=str(0), atype='pomcp',
                  position=(random_pos[0][0], random_pos[0][1]),
                  direction=random.sample(direction, 1)[0], radius=random.uniform(0.5, 1), angle=random.uniform(0.5, 1),
                  level=round(random.uniform(0.1, 1), 3)))

    # b. teammates and tasks
    LEVELS, MIN_COMBINATION = [], None
    for i in range(1, num_agents + num_tasks):
        if (i < num_agents):
            LEVELS.append(round(random.uniform(0.1, 1), 3))
            agents.append( \
                Agent(index=str(i), atype=random.sample(types, 1)[0], position=(random_pos[i][0], random_pos[i][1]),
                      direction=random.sample(direction, 1)[0], radius=random.uniform(0.5, 1),
                      angle=random.uniform(0.5, 1), level=LEVELS[-1]))
        else:
            if MIN_COMBINATION is None:
                if num_agents >= 4:
                    min_combination = [list(c) for c in list(itertools.combinations(LEVELS, 4))]
                    for j in range(len(min_combination)):
                        min_combination[j] = sum(min_combination[j])
                    min_combination = min(min_combination) if min(min_combination) <= 1 else 1.0
                else:
                    min_combination = 1.0

            sampleLevels = random.sample(LEVELS, 2)
            task_level = round(random.uniform(max(sampleLevels), sum(sampleLevels)), 3) \
                if sum(sampleLevels) <= 1 else round(random.uniform(max(sampleLevels), min_combination), 3)
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
    return env


def save_CaptureEnv(env, dim, num_agents, num_tasks, num_exp):
    file = open('./src/envs/maps/CaptureEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(num_exp) + '.pickle',
                'wb')
    env = pickle.dump(env, file)
    file.close()
    return env


def load_CaptureEnv(dim, num_agents, num_tasks, num_exp):
    with open('./src/envs/maps/CaptureEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(num_exp) + '.pickle',
              'rb') as map:
        env = pickle.load(map)
    return env


def create_CaptureEnv(dim, num_agents, num_tasks, partial_observable=False, display=False, num_exp=0):
    # 1. Importing the environment and its necessary components
    from src.envs.CaptureEnv import CaptureEnv, Agent, Task

    # 2. Defining the types and directions of the environment
    types = get_env_types('CaptureEnv')
    direction = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # 3. Getting the initial positions
    random_pos = get_initial_positions('CaptureEnv', dim, num_agents, num_tasks)

    # 4. Creating the components
    agents, tasks = [], []

    # a. main agent
    if not partial_observable:
        agents.append(
            Agent(index=str(0), atype='mcts',
                  position=(random_pos[0][0], random_pos[0][1]),
                  direction=random.sample(direction, 1)[0], radius=1.0, angle=1.0))
    else:
        agents.append(
            Agent(index=str(0), atype='pomcp',
                  position=(random_pos[0][0], random_pos[0][1]),
                  direction=random.sample(direction, 1)[0], radius=random.uniform(0.5, 1), angle=random.uniform(0.5, 1)))

    # b. teammates and tasks
    type_index = 0
    for i in range(1, num_agents + num_tasks):
        if (i < num_agents):
            agents.append( \
                Agent(index=str(i), atype=types[type_index % len(types)], position=(random_pos[i][0], random_pos[i][1]),
                      direction=random.sample(direction, 1)[0], radius=random.uniform(0.5, 1),
                      angle=random.uniform(0.5, 1)))
        else:
            tasks.append(Task(str(i), position=(random_pos[i][0], random_pos[i][1])))

        type_index += 1

    # c. adding to components dict
    components = {
        'agents': agents,
        'adhoc_agent_index': '0',
        'tasks': tasks}

    # 5. Initialising the environment and returning it
    if partial_observable:
        env = CaptureEnv((dim, dim), components, visibility='partial', display=display)
    else:
        env = CaptureEnv((dim, dim), components, visibility='full', display=display)

    save_CaptureEnv(env, dim, num_agents, num_tasks, num_exp)
    return env


"""
	SCENARIO GENERATOR - CREATION ROUTINE
"""
def creation_routine():
    """for dim in [20,25,30]:
        for num_agents in [5,7,10]:
            for num_tasks in [20,25,30]:
                for num_exp in range(100):
                    print('creating LFBEnv',num_agents,num_tasks,dim,num_exp)
                    create_LevelForagingEnv(dim, num_agents, num_tasks, num_exp=num_exp)"""
                    
    for setting in [[9,20,10]]:#[[5,5,10],[7,7,10],[10,10,10]]:
        for num_exp in range(100):
            print('creating CaptureEnv',setting[0],setting[1],setting[2],num_exp)
            create_CaptureEnv(setting[2], setting[0], setting[1], num_exp=num_exp)

import os
if not os.path.isdir("./src/envs/maps"):
    os.mkdir("./src/envs/maps")

if not os.path.isdir("./src/envs/maps/CaptureEnv"):
    os.mkdir("./src/envs/maps/CaptureEnv")

if not os.path.isdir("./src/envs/maps/LevelForagingEnv"):
    os.mkdir("./src/envs/maps/LevelForagingEnv")

from argparse import ArgumentParser, ArgumentTypeError

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0',''):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
if parser.prog == 'scenario_generator.py':
    parser.add_argument('--fixed', dest='fixed', default=False, type=str2bool,
                        help='Fixed maps: True or False')
    args = parser.parse_args()
    print(args)

    if args.fixed:
        creation_routine()