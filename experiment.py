from gym import spaces
import matplotlib.pyplot as plt
from src.reasoning.fundamentals import FundamentalValues
from src.reasoning.OEATA import *
from src.reasoning.parameter_estimation import *
from argparse import ArgumentParser
import sys
import numpy as np
import random
sys.path.append('src/reasoning')

from src.reasoning.estimation import *
from src.log import LogFile

from src.reasoning.AGA import *
from src.reasoning.ABU import *

parser = ArgumentParser()
parser.add_argument('--env', dest='env', default='CaptureEnv', type=str,
                    help='Environment name - LevelForagingEnv, CaptureEnv')
parser.add_argument('--estimation',dest='estimation',default='OEATA',type=str,help="Estimation type (AGA/ABU/OEATA) ")
parser.add_argument('--num_agents',dest='agents', default = 4, type = int, help = "Number of agents")
parser.add_argument('--num_tasks',dest='tasks',default=4,type=int,help = "Number of Tasks")
parser.add_argument('--dim',dest='dim',default=10,type=int,help="Dimension")
parser.add_argument('--num_exp',dest = 'num_exp',default=1,type=int,help='number of experiments')
parser.add_argument('--num_episodes',dest='num_episodes',type=int,default=10,help="number of episodes")
args = parser.parse_args()


if(args.env=="LevelForagingEnv"):
    from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task
elif(args.env=="CaptureEnv"):
    from src.envs.CaptureEnv import CaptureEnv, Agent, Task
else:
    print("Environment Does not exist")


def create_env(dim,num_agents,num_tasks):
    agents = []
    tasks = []
    if(args.env=="LevelForagingEnv"):
        types = ["l1","l2","l3"]
    else:
        types = ["c1","c2","c3"]
    direction = [0,np.pi/2,np.pi,3*np.pi/2]


    random_pos = random.sample([i for i in range(0, dim * dim)], num_agents + num_tasks)

    agents, tasks = [], []

    agents.append(
        Agent(index=str(0), atype='mcts',
              position=(random_pos[0] % dim, int(random_pos[0] / dim)),
              direction=random.sample(direction, 1)[0], radius=random.uniform(0.1, 1), angle=random.uniform(0.1, 1),
              level=random.uniform(0, 1)))

    for i in range(1, num_agents + num_tasks):
        if (i < num_agents):
            agents.append(
                Agent(index=str(i), atype=random.sample(types,1)[0], position=(random_pos[i] % dim, int(random_pos[i] / dim)),
                      direction=random.sample(direction,1)[0],radius=random.uniform(0.1,1), angle=random.uniform(0.1,1), level=random.uniform(0.1,1)))
        else:
            tasks.append(Task(str(i), position=(random_pos[i] % dim, int(random_pos[i] / dim)), level=random.uniform(0.1,1)))


    components = {
        'agents': agents,
        'adhoc_agent_index': '0',
        'tasks': tasks}
    if(args.env=="LevelForagingEnv"):
        env = LevelForagingEnv((dim, dim), components,visibility="full")
    else:
        env = CaptureEnv((dim,dim),components,visibility="full")

    return env

header = ["Iterations","Environment","Estimation","Actual Radius","Actual Angle","Actual Level", "Actual Types", "Radius Est.", "Angle Est.","Level Est.","Type Prob."]

def list_stats(env):

    stats = []
    iteration = env.episode
    environment = args.env
    estimation = args.estimation

    actual_radius = [a.radius for a in env.components['agents'] if a.index != '0']
    actual_angle = [a.angle for a in env.components['agents'] if a.index != '0']
    actual_level = [a.level for a in env.components['agents'] if a.index != '0']
    actual_type = [a.type for a in env.components['agents'] if a.index != '0']

    est_radius,est_angle,est_level,type_prob = [],[],[],[]

    for a in env.components['agents']:
        if(a.index == '0'):
            continue

        selected_type = a.smart_parameters['estimations'].get_highest_type_probability()
        selected_parameter = a.smart_parameters['estimations'].get_parameters_for_selected_type(selected_type)
        est_radius.append(selected_parameter.radius)
        est_angle.append(selected_parameter.angle)
        est_level.append(selected_parameter.level)
        type_prob.append(a.smart_parameters['estimations'].get_probability_type(a.type))


    stats.append(iteration)
    stats.append(environment)
    stats.append(estimation)
    stats.append(actual_radius)
    stats.append(actual_angle)
    stats.append(actual_level)
    stats.append(actual_type)
    stats.append(est_radius)
    stats.append(est_angle)
    stats.append(est_level)
    stats.append(type_prob)

    return stats


for exp in range(1,args.num_exp+1):
    fname = "./results/{}_a{}_i{}_dim{}_{}_exp{}.csv".format(args.env,args.agents,args.tasks,args.dim,args.estimation,exp)
    log_file = LogFile(None,fname,header)

    if(args.env=="LevelForagingEnv"):
        # Estimator Configuration
        estimation_mode = args.estimation
        oeata_parameter_calculation_mode = 'MEAN'  # It can be MEAN, MODE, MEDIAN
        agent_types = ['l1', 'l2', 'l3']
        estimators_length = 100
        mutation_rate = 0.2
        aga, abu = None, None

        fundamental_values = FundamentalValues(radius_max=1, radius_min=0.1, angle_max=1, angle_min=0.1, level_max=1,
                                               level_min=0, agent_types=agent_types, env_dim=[args.dim, args.dim],
                                               estimation_mode=estimation_mode)

    else:
        # Estimator Configuration
        estimation_mode = args.estimation
        oeata_parameter_calculation_mode = 'MEAN'  # It can be MEAN, MODE, MEDIAN
        agent_types = ['c1', 'c2', 'c3']
        estimators_length = 100
        mutation_rate = 0.2
        aga, abu = None, None

        fundamental_values = FundamentalValues(radius_max=1, radius_min=0.1, angle_max=1, angle_min=0.1, level_max=1,
                                               level_min=0, agent_types=agent_types, env_dim=[args.dim, args.dim],
                                               estimation_mode=estimation_mode)

    env = create_env(args.dim,args.agents,args.tasks)
    state = env.reset()

# 4 is the dimension of the grid
    if estimation_mode == 'AGA':

        estimation_config = AGAConfig(fundamental_values, 4, 0.01, 0.999)
        aga = AGAprocess(estimation_config, env)

    elif estimation_mode == "ABU":
        estimation_config = ABUConfig(fundamental_values, 4)
        abu = ABUprocess(estimation_config, env)

    adhoc_agent = env.get_adhoc_agent()

    if estimation_mode == 'OEATA':
        estimation_config = OeataConfig(estimators_length, oeata_parameter_calculation_mode, mutation_rate,
                                                 fundamental_values)

        for a in env.components['agents']:
            a.smart_parameters['last_completed_task'] = None
            a.smart_parameters['choose_task_state'] = env.copy()
            if a.index != adhoc_agent.index:
                param_estim = ParameterEstimation(estimation_config)
                param_estim.estimation_initialisation()
                oeata = OEATA_process(estimation_config, a)
                oeata.initialisation(a.position, a.direction, a.radius,a.angle,a.level, env)
                param_estim.learning_data = oeata
                a.smart_parameters['estimations'] = param_estim

    done = False
    while not done and env.episode < args.num_episodes:
        # Rendering the environment
#        env.render()

        # Main Agent taking an action
        module = __import__(adhoc_agent.type)
        method = getattr(module, adhoc_agent.type+'_planning')

        if(adhoc_agent.type == "mcts" or adhoc_agent.type=="pomcp"):
            adhoc_agent.next_action, adhoc_agent.target = method(state,adhoc_agent)
        else:
            adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)

        if (estimation_mode == 'AGA'):
            aga.update(state)
        elif (estimation_mode == 'ABU'):
            abu.update(state)


        # Step on environment
        state, reward, done, info = env.step(adhoc_agent.next_action)
        just_finished_tasks = info['just_finished_tasks']
        # Verifying the end condition
        print(len(just_finished_tasks))
        if (estimation_mode == 'OEATA'):
            if(args.env=="LevelForagingEnv"):
                level_foraging_uniform_estimation(env, just_finished_tasks)
            else:
                print(1)
                capture_uniform_estimation(env,just_finished_tasks)

        stats = list_stats(env)
        print(stats)
        log_file.write(None, stats)

        if done:
            break



