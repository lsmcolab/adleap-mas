###
# Imports
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.TigerEnv import load_default_scenario

from argparse import ArgumentParser, ArgumentTypeError

###
# Support method
###
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

###
# Setting the environment
###
# 1. Reading the environment settings
parser = ArgumentParser()
# default args
parser.add_argument('--exp_num', dest='exp_num', default=0, type=int)
parser.add_argument('--atype', dest='atype', default='pomcp', type=str)
args = parser.parse_args()
print('|||||||||||',args)

###
# TIGER ENVIRONMENT SETTINGS
###
header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
log = LogFile('TigerEnv',0,args.atype,args.exp_num,header)

round, MAX_ROUNDS = 0, 50
MAX_EPISODES = 20
###
# ADLEAP-MAS MAIN ROUTINE
###
while round < MAX_ROUNDS:
    # env components and settings
    env, scenario_id = load_default_scenario(args.atype,0)
    
    state = env.reset()
    agent = env.get_adhoc_agent()

    # running the tiger problem
    done = False
    while env.episode < MAX_EPISODES and not done:
        # 1. Importing agent method
        method = env.import_method(agent.type)

        # 2. Reasoning about next action and target
        start = time.time()
        agent.next_action, _ = method(state, agent)
        end = time.time()

        # 3. Taking a step in the environment
        next_state, reward, done, _ = env.step(action=agent.next_action)

        data = {'it':env.episode,
                'reward':reward,
                'time':end-start,
                'nrollout':agent.smart_parameters['count']['nrollouts'],
                'nsimulation':agent.smart_parameters['count']['nsimulations']}
        log.write(data)
        state = next_state
        
    round += 1
    env.close()
###
# THE END - That's all folks :)
###