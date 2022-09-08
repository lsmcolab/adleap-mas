##
## Main code: template to run a single (default) experiment on AdLeap-MAS
##
# 1. Setting the environment
method = 'pomcp'                # choose your method
kwargs = {}                     # define your additional hyperparameters to it (optional)
env_name = 'LevelForagingEnv'   # choose your environment
scenario_id = 1                 # define your scenario configuration (check the available configuration in our GitHub)
estimation_method = 'aga'       # choosing your estimation method
estimation_args = None          # don't need to change (loaded afterwards)
display = False                 # choosing to turn on or off the display


# 2. Creating the environment
# a. importing necessary modules
from importlib import import_module
from src.log import LogFile
import time

# b. creating the environment
env_module = import_module('src.envs.'+env_name)
load_default_scenario_method = getattr(env_module, 'load_default_scenario')
env, scenario_id = load_default_scenario_method(method,scenario_id,display=display)

# 3. Creating Helper Functions
def get_env_types(env_name):
    if env_name == "LevelForagingEnv":
        return ['l1', 'l2']#, 'l3', 'l4', 'l5', 'l6']
    elif env_name == "CaptureEnv":
        return ['c1', 'c2'] # , 'c3'
    else:
        raise NotImplemented

def get_env_parameters_minmax(env_name):
    if env_name == "LevelForagingEnv":
        return [(0.5,1),(0.5,1),(0.5,1)]
    elif env_name == "CaptureEnv":
        return [(0.5,1),(0.5,1)]
    else:
        raise NotImplemented

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
adhoc_agent = env.get_adhoc_agent()
adhoc_agent.smart_parameters['estimation_method'] = estimation_method
adhoc_agent.smart_parameters['estimation_args'] =\
     get_env_types(env_name),get_env_parameters_minmax(env_name)

exp_num = 0
header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
log = LogFile(env_name,scenario_id,method,exp_num,header)

MAX_EPISODES = 200
done = False
while not done and env.episode < MAX_EPISODES:
    # 1. Importing agent method
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    start = time.time()
    adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent,kwargs=kwargs)
    end = time.time()

    # 3. Taking a step in the environment
    state,reward,done,info = env.step(adhoc_agent.next_action)

    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations']}
    log.write(data)

    # 4. Printing the estimation
    print('Episode:', env.episode,' | Estimation: ')
    adhoc_agent.smart_parameters['estimation'].show_estimation(env)

env.close()
###
# THE END - That's all folks :)
###
