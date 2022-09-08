###
# Imports
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.LevelForagingEnv import load_default_scenario

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
# additional args
parser.add_argument('--id', dest='id', default=0, type=int) # scenario id
args = parser.parse_args()
print('|||||||||||',args)

# 2. Creating the environment
env, scenario_id = load_default_scenario(args.atype,scenario_id=args.id,display=False)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
agent = env.get_adhoc_agent()

header = ['Iteration','Reward','Time to reason','N Rollouts', 'N Simulations']
log = LogFile('LevelForagingEnv',scenario_id,args.atype,args.exp_num,header)

MAX_EPISODES = 200
done = False
while not done and env.episode < MAX_EPISODES:
    # 1. Importing agent method
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    start = time.time()
    agent.next_action, _ = method(state, agent)
    end = time.time()
    
    # 3. Taking a step in the environment
    state,reward,done,info = env.step(agent.next_action)
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start,
            'nrollout':agent.smart_parameters['count']['nrollouts'],
            'nsimulation':agent.smart_parameters['count']['nsimulations']}
    log.write(data)

env.close()
###
# THE END - That's all folks :)
###