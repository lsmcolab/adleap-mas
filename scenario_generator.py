import itertools
import numpy as np
import os
import pickle
import random

def get_initial_positions(env_name, dim, nagents, ntasks):
    # verifying if it is possible to create the scenario
    if env_name == "LevelForagingEnv" and (ntasks * 9) > (dim ** 2):
        print('It is not possible to create the '+ env_name+' scenario with'+\
            str(ntasks)+' and '+str(dim)+'-- not enough room to it.\n We reduced the'+\
                ' number of tasks from '+str(ntasks)+' to '+str(int((dim**2)/9) - 1)+'.')
        ntasks = int((dim**2)/9) - 1

    # getting the random positions
    pos = []
    while len(pos) < (nagents + ntasks):
        x = random.randint(0, dim - 1)
        y = random.randint(0, dim - 1)

        if env_name == "LevelForagingEnv":
            if len(pos) <= ntasks:
                if x > 0 and x < dim-1 and y > 0 and y < dim-1 and \
                 (x, y) not in pos and (x + 1, y) not in pos and \
                 (x + 1, y + 1) not in pos and (x, y + 1) not in pos and \
                 (x - 1, y + 1) not in pos and (x - 1, y) not in pos and \
                 (x - 1, y - 1) not in pos and (x, y - 1) not in pos and \
                 (x + 1, y - 1) not in pos:
                    pos.append((x, y))
            elif (x, y) not in pos:
                pos.append((x, y))
        elif env_name == "CaptureEnv":
            if (x, y) not in pos:
                pos.append((x, y))
        else:
            raise NotImplemented

    pos.reverse()
    if env_name == "LevelForagingEnv":
        return pos, ntasks
    return pos


def get_env_types(env_name):
    if env_name == "LevelForagingEnv":
        return ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
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

def get_env_parameters_minmax(env_name):
    if env_name == "LevelForagingEnv":
        return [(0.5,1),(0.5,1),(0.5,1)]
    elif env_name == "CaptureEnv":
        return [(0.5,1),(0.5,1)]
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
    print('Loading Level Foraging Env #'+str(num_exp)+':', dim, num_agents, num_tasks)
    map_path = './src/envs/maps/LevelForagingEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(num_exp) + '.pickle'
    if os.path.isfile(map_path):
        with open(map_path, 'rb') as map:
            env = pickle.load(map)
    else:
        raise FileNotFoundError
    return env

def create_LevelForagingEnv(dim, num_agents, num_tasks, partial_observable=False, display=False, num_exp=0):
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
    return env


def save_CaptureEnv(env, dim, num_agents, num_tasks, num_exp):
    file = open('./src/envs/maps/CaptureEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(num_exp) + '.pickle',
                'wb')
    env = pickle.dump(env, file)
    file.close()
    return env


def load_CaptureEnv(dim, num_agents, num_tasks, num_exp):
    print('Loading Capture Env #'+str(num_exp)+':', dim, num_agents, num_tasks)
    map_path = './src/envs/maps/CaptureEnv/' + str(dim) + str(num_agents) + str(num_tasks) + str(num_exp) + '.pickle'
    if os.path.isfile(map_path):
        with open(map_path,'rb') as map:
            env = pickle.load(map)
    else:
        raise FileNotFoundError
    return env


def create_CaptureEnv(dim, num_agents, num_tasks, partial_observable=False, display=False, num_exp=0):
    print('Creating Capture Env #'+str(num_exp)+':', dim, num_agents, num_tasks)
    # 1. Importing the environment and its necessary components
    from src.envs.CaptureEnv import CaptureEnv, Agent, Task

    # 2. Defining the types and directions of the environment
    types = get_env_types('CaptureEnv')
    direction = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # 3. Getting the initial positions
    random_pos = get_initial_positions('CaptureEnv', dim, num_agents, num_tasks)

    # 4. Creating the components 
    # ####
    # SETTINGS INIT
    ####
    agents, tasks = [], []

    MIN_RADIUS, MAX_RADIUS = 0.5, 1.0
    MIN_ANGLE, MAX_ANGLE = 0.5, 1.0

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
                  direction=random.sample(direction, 1)[0], 
                  radius=random.uniform(MIN_RADIUS, MAX_RADIUS), angle=random.uniform(MIN_ANGLE, MAX_ANGLE)))

    # b. teammates and tasks
    type_index = 0
    for i in range(1, num_agents + num_tasks):
        if (i < num_agents):
            agents.append( \
                Agent(index=str(i), atype=types[type_index % len(types)], position=(random_pos[i][0], random_pos[i][1]),
                      direction=random.sample(direction, 1)[0], radius=random.uniform(MIN_RADIUS, MAX_RADIUS),
                      angle=random.uniform(MIN_ANGLE, MAX_ANGLE)))
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