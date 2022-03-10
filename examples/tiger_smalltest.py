###
# IMPORTS
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.TigerEnv import TigerEnv, Agent

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
# default args
parser.add_argument('--exp_num', dest='exp_num', default=0, type=int)
parser.add_argument('--atype', dest='atype', default='pomcp', type=str)
args = parser.parse_args()
print('|||||||||||',args)

###
# TIGER ENVIRONMENT SETTINGS
###
header = ['Iteration','Reward','Time to reason']
log = LogFile('TigerEnv','tiger_'+args.atype+'_test_'+str(args.exp_num)+'.csv',header)

round, MAX_ROUNDS = 0, 10
MAX_EPISODES = 40
while round < MAX_ROUNDS:
    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    agent = Agent(0,args.atype)
    components = {"agents":[agent]}
    cum_rwd, discount_factor = 0.0, 0.75
    display = False

    env = TigerEnv(components=components,tiger_pos='left',display=display)  
    state = env.reset()


    done = False
    while env.episode < MAX_EPISODES and not done:
        #env.render()
            
        # 1. Importing agent method
        agent = env.get_adhoc_agent()
        method = env.import_method(agent.type)

        # 2. Reasoning about next action and target
        start = time.time()
        agent.next_action, _ = method(state, agent)
        end = time.time()

        # 3. Taking a step in the environment
        next_state, reward, done, _ = env.step(action=agent.next_action)

        data = {'it':env.episode,
                'reward':reward,
                'time':end-start}
        log.write(None,data)
        state = next_state
        
    env.close()
###
# THE END - That's all folks :)
###