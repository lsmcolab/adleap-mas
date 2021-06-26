"""
 Code to run experiments in AdLeap-MAS
 - Sections:
    A. IMPORTS
    B. ARGS PARSE
    C. AUX FUNCTIONS
    D. MAIN SCRIPT
"""
###
# A. IMPORTS
###
import numpy as np
import sys
sys.path.append('src/reasoning')

from scenario_generator import *
from src.reasoning.estimation import aga_estimation, abu_estimation, oeata_estimation
from src.log import LogFile

###
# B. ARGS PARSE
###
# Getting the experiment setup via argument parsing
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--env', dest='env', default='LevelForagingEnv', type=str,
                    help='Environment name - LevelForagingEnv, CaptureEnv')
parser.add_argument('--estimation',dest='estimation',default='AGA',type=str,help="Estimation type (AGA/ABU/OEATA) ")
parser.add_argument('--num_agents',dest='agents', default = 5, type = int, help = "Number of agents")
parser.add_argument('--num_tasks',dest='tasks',default=10,type=int,help = "Number of Tasks")
parser.add_argument('--dim',dest='dim',default=10,type=int,help="Dimension")
parser.add_argument('--num_exp',dest = 'num_exp',default=1,type=int,help='number of experiments')
parser.add_argument('--num_episodes',dest='num_episodes',type=int,default=200,help="number of episodes")
parser.add_argument('--po',dest='po',type=bool,default=False,help="Partial Observability (True/False) ")
parser.add_argument('--display',dest='display',type=bool,default=False,help="Display (True/False) ")
args = parser.parse_args()

###
# C. AUX FUNCTIONS
###
def list_stats(env):
    stats = []
    iteration = env.episode
    environment = args.env
    estimation = args.estimation

    actual_radius = [a.radius for a in env.components['agents'] if a.index != '0']
    actual_angle = [a.angle for a in env.components['agents'] if a.index != '0']
    actual_level = [a.level for a in env.components['agents'] if a.index != '0']
    actual_type = [a.type for a in env.components['agents'] if a.index != '0']

    adhoc_agent = env.get_adhoc_agent()
    _, type_probs, param_est =\
        adhoc_agent.smart_parameters['estimation'].get_estimation(env)
        
    completion = sum([t.completed for t in env.components['tasks']])/len(env.components['tasks'])
    print(iteration,'-',environment,'-',estimation,'-',completion,'%')
    
    type_probabilities, est_radius, est_angle, est_level = [], [], [], []
    for teammate in env.components['agents']:
        if teammate.index != adhoc_agent.index:
            type_probabilities.append(type_probs[teammate.index])
            est_radius.append(param_est[teammate.index][0])
            est_angle.append(param_est[teammate.index][1])
            est_level.append(param_est[teammate.index][2])
        
    print("| Act.Type:",actual_type)
    for agent_type_prob in type_probabilities:
        print("| ",agent_type_prob)
    print("| Act.Radius:\n| ", actual_radius,"\n| ",est_radius)
    print("| Act.Angle:\n| ",actual_angle,"\n| ",est_angle)
    print("| Act.Level:\n| ",actual_level,"\n| ",est_level)

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
    stats.append(type_probabilities)

    return stats

###
# D. MAIN SCRIPT
###1. Initialising the log file
header = ["Iterations","Environment","Estimation","Actual Radius","Actual Angle","Actual Level", "Actual Types", "Radius Est.", "Angle Est.","Level Est.","Type Prob."]
fname = "./results/{}_a{}_i{}_dim{}_{}_exp{}.csv".format(args.env,args.agents,args.tasks,args.dim,args.estimation,args.num_exp)
log_file = LogFile(None,fname,header)

# 2. Creating the environment
if args.env == 'LevelForagingEnv':
    env = create_LevelForagingEnv(args.dim,args.agents,args.tasks,args.po,args.display)
elif args.env == 'CaptureEnv':
    env = create_CaptureEnv(args.dim,args.agents,args.tasks,args.po,args.display)
else:
    raise NotImplemented
state = env.reset()

# 3. Estimation algorithm's settings
estimation_mode = args.estimation
adhoc_agent = env.get_adhoc_agent()

if args.estimation == 'AGA':
    adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(args.env), [(0,1),(0,1),(0,1)]
    estimation_method = aga_estimation
elif  args.estimation == 'ABU':
    adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(args.env), [(0,1),(0,1),(0,1)]
    estimation_method = abu_estimation
elif args.estimation == 'OEATA':
    adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(args.env), get_env_nparameters(args.env), 100, 2, 0.2, 100, np.mean
    estimation_method = oeata_estimation
else:
    raise NotImplemented

# 4. Starting the experiment
done = False
print(args.env," Visibility:",env.visibility, " Display:",env.display)
while not done and env.episode < args.num_episodes:
    # Rendering the environment
    if env.display:
        env.render()
    print("Episode : ", env.episode)

    # Main Agent taking an action
    module = __import__(adhoc_agent.type)
    method = getattr(module, adhoc_agent.type+'_planning')
    adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent, estimation_algorithm=estimation_method)

    for ag in env.components['agents']:
        print(ag.index,ag.target)

    # Step on environment
    state, reward, done, info = env.step(adhoc_agent.next_action)
    just_finished_tasks = info['just_finished_tasks']

    # Writing log
    stats = list_stats(env)
    log_file.write(None, stats)

    # Verifying the end condition
    if done:
        break
