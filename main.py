##
## Main code: template to run a single (default) experiment on AdLeap-MAS
##
# 1. Setting the environment
method = 'pomcp'                # choose your method
kwargs = {}                     # define your additional hyperparameters to it (optional)
env_name = 'LevelForagingEnv'   # choose your environment
scenario_id = 0                 # define your scenario configuration (check the available configuration in our GitHub)

display = True                  # choosing to turn on or off the display

# 2. Creating the environment
# a. importing necessary modules
from importlib import import_module
from src.log import LogFile
import time

# b. creating the environment
env_module = import_module('src.envs.'+env_name)
load_default_scenario_method = getattr(env_module, 'load_default_scenario')
env, scenario_id = load_default_scenario_method(method,scenario_id,display=display)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
adhoc_agent = env.get_adhoc_agent()

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
    next_state,reward,done,info = env.step(adhoc_agent.next_action)

    # 4. For learning methods
    if 'dqn_model' in adhoc_agent.smart_parameters.keys():
        adhoc_agent.smart_parameters['dqn_model'].add_memory(state,adhoc_agent.next_action,next_state,reward,done)

    # 5. Logging the Data
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':adhoc_agent.smart_parameters['count']['nrollouts'],
            'nsimulation':adhoc_agent.smart_parameters['count']['nsimulations']}
    log.write(data)
    
    # 6. Updating the state
    state = next_state

env.close()
###
# THE END - That's all folks :)
###
