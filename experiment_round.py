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
from src.reasoning.estimation import aga_estimation, abu_estimation, oeate_estimation, pomcp_estimation
from src.log import BashLogFile, LogFile

###
# B. ARGS PARSE
###
# Getting the experiment setup via argument parsing
from argparse import ArgumentParser, ArgumentTypeError

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
parser.add_argument('--env', dest='env', default='LevelForagingEnv', type=str,
                    help='Environment name - LevelForagingEnv, CaptureEnv')
parser.add_argument('--estimation',dest='estimation',default='OEATE',type=str,help="Estimation type (AGA/ABU/OEATE/POMCP) ")
parser.add_argument('--num_agents',dest='agents', default = 7, type = int, help = "Number of agents")
parser.add_argument('--num_tasks',dest='tasks',default=20,type=int,help = "Number of Tasks")
parser.add_argument('--dim',dest='dim',default=20,type=int,help="Dimension")
parser.add_argument('--num_exp',dest = 'num_exp',default=0,type=int,help='number of experiments')
parser.add_argument('--num_episodes',dest='num_episodes',type=int,default=200,help="number of episodes")
parser.add_argument('--po',dest='po',type=str2bool,default=False,help="Partial Observability (True/False) ")
parser.add_argument('--display',dest='display',type=str2bool,nargs='?',const=True,default=False,help="Display (True/False) ")
args = parser.parse_args()

print(args)

###
# C. AUX FUNCTIONS
###
def list_stats(env, accomplished_tasks):
    stats = {}
    stats['iteration'] = env.episode
    stats['completion'] = accomplished_tasks
    stats['environment'] = args.env
    stats['estimation'] = args.estimation
    stats['actual_radius'] = [a.radius for a in env.components['agents'] if a.index != '0']
    stats['actual_angle'] = [a.angle for a in env.components['agents'] if a.index != '0']
    if args.env == "LevelForagingEnv":
        stats['actual_level'] = [a.level for a in env.components['agents'] if a.index != '0']
    else:
        stats['actual_level'] = np.zeros(len(env.components['agents'])-1)
    stats['actual_type'] = [a.type for a in env.components['agents'] if a.index != '0']

    adhoc_agent = env.get_adhoc_agent()
    type_probabilities, estimated_parameters =\
        adhoc_agent.smart_parameters['estimation'].get_estimation(env)

    stats['est_radius'], stats['est_angle'], stats['est_level'] = [], [], []
    for i in range(len(env.components['agents'])-1):
        stats['est_radius'].append([estimated_parameters[i][j][0] for j in range(len(adhoc_agent.smart_parameters['estimation'].template_types))])
        stats['est_angle'].append([estimated_parameters[i][j][1] for j in range(len(adhoc_agent.smart_parameters['estimation'].template_types))])
        if args.env == 'LevelForagingEnv':
            stats['est_level'].append([estimated_parameters[i][j][2] for j in range(len(adhoc_agent.smart_parameters['estimation'].template_types))])
        else:
            stats['est_level'].append(list(np.zeros(len(adhoc_agent.smart_parameters['estimation'].template_types))))
    stats['type_probabilities'] = type_probabilities
    
    return stats

###
# D. MAIN SCRIPT
###1. Initialising the log file
header = ["Iterations","Completion","Environment","Estimation","Actual Radius","Actual Angle","Actual Level", "Actual Types", "Radius Est.", "Angle Est.","Level Est.","Type Prob."]
fname = "Round_{}_a{}_i{}_dim{}_{}_exp{}.csv".format(args.env,args.agents,args.tasks,args.dim,args.estimation,args.num_exp)
log_file = LogFile(None,fname,header)
bashlog_file = BashLogFile(fname)

# 2. Creating the environment
env = None
if os.path.isdir("./src/envs/maps"):
    if os.path.isdir("./src/envs/maps/"+args.env):
        map_path = './src/envs/maps/'+args.env +'/' + str(args.dim) + str(args.agents) +\
             str(args.tasks) + str(args.num_exp) + '.pickle'
        if os.path.isfile(map_path):
            if args.env == 'LevelForagingEnv':
                env = load_LevelForagingEnv(args.dim,args.agents,args.tasks,args.num_exp)
            elif args.env == 'CaptureEnv':
                env = load_CaptureEnv(args.dim,args.agents,args.tasks,args.num_exp)
            else:
                raise NotImplemented
if env is None:
    if args.env == 'LevelForagingEnv':
        env = create_LevelForagingEnv(args.dim,args.agents,args.tasks,args.po,args.display, args.num_exp)
    elif args.env == 'CaptureEnv':
        env = create_CaptureEnv(args.dim,args.agents,args.tasks,args.po,args.display, args.num_exp)
    else:
        raise NotImplemented

state = env.reset()

# 3. Estimation algorithm's settings
estimation_mode = args.estimation
adhoc_agent = env.get_adhoc_agent()

if args.estimation == 'AGA':
    adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(args.env), get_env_parameters_minmax(args.env)

    estimation_method = aga_estimation

elif  args.estimation == 'ABU':
    adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(args.env), get_env_parameters_minmax(args.env)

    estimation_method = abu_estimation

elif args.estimation == 'OEATE':
    adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(args.env), get_env_parameters_minmax(args.env),\
      100, 2, 0.2, 100, np.mean

    estimation_method = oeate_estimation
    
elif args.estimation == 'POMCP':
    adhoc_agent.smart_parameters['estimation_args'] =\
    get_env_types(args.env), get_env_parameters_minmax(args.env)

    estimation_method = pomcp_estimation

else:
    estimation_method = None

# 4. Starting the experiment
done = False
exp_round = 0
tasks_per_round = int(args.tasks/2) if int(args.tasks/2) > 0 else 1
env.display = args.display
print(args.env," Visibility:",env.visibility, " Display:",env.display)

for i in range(len(env.components['tasks'])):
    if i != (tasks_per_round*exp_round) % len(env.components['tasks']) and\
     i != (tasks_per_round*exp_round+1) % len(env.components['tasks']):
        env.components['tasks'][i].completed = True
    else:
        env.components['tasks'][i].completed = False

accomplished_tasks = 0

###
# EXPERIMENT START
###
bashlog_file.redirect_stderr()

while env.episode < args.num_episodes:
    # Rendering the environment
    if env.display:
        env.render()
    print("Episode : "+str(env.episode))

    # Main Agent taking an action
    print("Main Agent planning")
    module = __import__(adhoc_agent.type)
    method = getattr(module, adhoc_agent.type+'_planning')
    adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent, estimation_algorithm=estimation_method)

    if env.episode == 0:
        stats = list_stats(env, accomplished_tasks)
        log_file.write(None, stats)

    # Step on environment
    print("Simulation Step")
    state, reward, done, info = env.step(adhoc_agent.next_action)
    just_finished_tasks = info['just_finished_tasks']
    accomplished_tasks += len(just_finished_tasks)

    # Writing log
    print("Log\n")
    stats = list_stats(env, accomplished_tasks)
    log_file.write(None, stats)

    # Verifying the end condition
    if done:
        exp_round += 1
        for i in range(len(env.components['tasks'])):
            if i != (tasks_per_round*exp_round) % len(env.components['tasks']) and\
             i != (tasks_per_round*exp_round+1) % len(env.components['tasks']):
                env.components['tasks'][i].completed = True
            else:
                env.components['tasks'][i].completed = False

bashlog_file.reset_stderr()
###
# EXPERIMENT END
###
