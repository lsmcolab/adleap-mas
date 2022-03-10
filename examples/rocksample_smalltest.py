###
# IMPORTS
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.RockSampleEnv import RockSampleEnv, Rock, Agent

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
# ROCK SAMPLE ENVIRONMENT SETTINGS
###
rock_1 = Rock(0,(2,2),"good")
rock_2 = Rock(1,(5,2),"good")
rock_3 = Rock(2,(3,7),"bad")
rock_4 = Rock(3,(6,8),"bad")
agent = Agent(0,(3,3),args.atype)
components = {"rocks":[rock_1,rock_2,rock_3,rock_4],"agents":[agent]}
display = False
MAX_EPISODES = 200

###
# ADLEAP-MAS MAIN ROUTINE
###
env = RockSampleEnv(components=components,dim=10,display=display)
state = env.reset()

header = ['Iteration','Reward','Time to reason']
log = LogFile('RockSampleEnv','rocksample_'+args.atype+'_test_'+str(args.exp_num)+'.csv',header)

done = False
while env.episode < MAX_EPISODES and not done:
    #env.render()
        
    # 1. Importing agent method
    agent = env.get_adhoc_agent()
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    start = time.time()
    if agent.type == 'pomcp' or agent.type=='mcts':
        agent.next_action, _ = method(state, agent, 
            black_box_simulation=True,particle_revigoration=True)
    else:
        agent.next_action, _ = method(state, agent)
    end = time.time()

    # 3. Taking a step in the environment
    state,reward,done,info = env.step(agent.next_action)
    data = {'it':env.episode,
            'reward':reward,
            'time':end-start}
    log.write(None,data)
    
env.close()
###
# THE END - That's all folks :)
###