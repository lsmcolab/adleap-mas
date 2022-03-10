###
# IMPORTS
###
import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.LevelForagingEnv import LevelForagingEnv,Agent,Task

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
# additional args
parser.add_argument('--bb',dest='bb',default='True',type=str2bool)
parser.add_argument('--pr',dest='pr',default='True',type=str2bool)
args = parser.parse_args()
print('|||||||||||',args)

###
# LEVEL-BASED FORAGING ENVIRONMENT SETTINGS
###
components = {
    'agents' : [
            Agent(index='A',atype=args.atype,position=(1,1),direction=1*np.pi/2,radius=0.5,angle=1,level=1.0), 
                ],
    'adhoc_agent_index' : 'A',
    'tasks' : [
            Task(index='0',position=(8,8),level=1.0),
            Task(index='1',position=(5,5),level=1.0),
            Task(index='2',position=(0,0),level=1.0),
            Task(index='3',position=(9,1),level=1.0),
            Task(index='3',position=(0,9),level=1.0)
                ]
}

dim = (10,10)
display = False
visibility = 'partial'
estimation_method = None

MAX_TIME = 2*60 # 2 minutes

###
# ADLEAP-MAS MAIN ROUTINE
###
env = LevelForagingEnv(shape=dim,components=components,visibility=visibility,display=display)
state = env.reset()
adhoc_agent = env.get_adhoc_agent()

header = ['Iteration','N Completed Tasks','Time to reason']
log = LogFile('LevelForagingEnv','levelforaging_'+args.atype+'_test_'+str(args.exp_num)+'.csv',header)

done = False   
exp_start = time.time()
while not done and int(time.time() - exp_start) < MAX_TIME:
    #env.render(sleep_=0.0)

    # 1. Importing agent method
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    start = time.time()
    if args.atype == 'pomcp':
        adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent,
            black_box_simulation=args.bb,particle_revigoration=args.pr)
    else:
        adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)
    end = time.time()

    # 3. Taking a step in the environment
    state,reward,done,info = env.step(adhoc_agent.next_action)

    data = {'it':env.episode,
            'ntasks':sum([task.completed for task in env.components['tasks']]),
            'time':end-start}
    log.write(None,data)

env.close()
###
# THE END - That's all folks :)
###