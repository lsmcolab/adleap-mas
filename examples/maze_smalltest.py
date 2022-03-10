###
# IMPORTS
###
import sys
import os
import time

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.MazeEnv import MazeEnv, Agent

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
parser.add_argument('--id', dest='id', default=0, type=int) # scenario id
args = parser.parse_args()
print('|||||||||||',args)

###
# MAZE ENVIRONMENT SETTINGS
###
scenario = [{"dim":5,"black":[(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(3,4),(4,4)]},
            {"dim":8,"black":[(0,1),(0,3),(0,5),(0,7),(2,1),(2,3),(2,5),(2,7),(6,1),(6,3),(6,5),(6,7),(4,1),(4,5),(4,7)]},
            {"dim":6,"black":[(1,1),(1,4),(4,1)]},
            {"dim":3,"black":[(0,0),(1,0),(2,0),(1,1),(2,1),(2,2)]}]

scenario_id = args.id

agent = Agent(index= 0,
        position= (int(scenario[scenario_id]["dim"]/2),int(scenario[scenario_id]["dim"]/2)),
        type= args.atype)
components = {"agents":[agent],
    "black":scenario[scenario_id]["black"]}
display = False
MAX_EPISODES = 200

###
# ADLEAP-MAS MAIN ROUTINE
###
env = MazeEnv(components=components,dim=scenario[scenario_id]["dim"],display=display)
state = env.reset()

header = ['Iteration','Reward','Time to reason']
log = LogFile('MazeEnv','maze.'+str(scenario_id)+'/maze_'+args.atype+'_test_'+str(args.exp_num)+'.csv',header)

done = False  
while not done and env.episode < MAX_EPISODES:
    #env.render(sleep_=0.0)

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