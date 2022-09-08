###
# IMPORTS
###
import time
import sys
import os
import numpy as np
from multiprocessing import freeze_support

sys.path.append(os.getcwd())

from src.log import LogFile
from src.envs.SmartFireBrigadeEnv import SmartFireBrigadeEnv, Agent, Fire

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
parser.add_argument('--atype', dest='atype', default='scouter', type=str)
parser.add_argument('--mode', dest='mode', default='control', type=str)
args = parser.parse_args()
print('|||||||||||',args)

###
# SMART FIRE BRIGADE ENVIRONMENT SETTINGS
###
TIMEOUT = 5*60 # 5 minutes

# Main routine
def main(args):
    global TIMEOUT

    # Creating the environment and defining its components
    components = {
        'agents':[\
            Agent(index='A'+str(args.exp_num)+'_'+str(args.atype),\
                    atype=args.atype,radius=15,angle=np.pi/2)
            ],
        'fire':[\
            Fire(position=(10,15), level=3, time_constraint=True),\
            Fire(position=(20,25), level=3, time_constraint=True),\
            Fire(position=(29,15), level=3, time_constraint=True),\
                ]
        }

    for a in components['agents']:
        try:
            os.remove('./tmp/'+a.index+'.txt')
        except:
            pass

    env = SmartFireBrigadeEnv(components=components,dim=(30,30),action_mode=args.mode,display=True)
    state = env.reset()

    header = ['Time','Reward','FireSpreadingLevels','Actions','WaterLeft','BatterLeft']
    log = LogFile('SmartFireBrigade','smartfirebrigade_'+args.atype+'_test_'+str(args.exp_num)+'.csv',header)

    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    done = False    
    start_time = time.time()
    total_reward = 0
    steps_until_write = 0
    write_time = time.time()

    actions_to_write = {}
    for agent in env.components['agents']:
        actions_to_write[agent.index] = []

    while not done and (time.time()  - start_time) < TIMEOUT:
        if env.display:
            env.render()

        actions = env.get_step_actions()
        state,reward,done,info = env.step(actions)
        for a in actions:
            if len(actions_to_write[a]) == 0:
                actions_to_write[a].append(actions[a])
            elif actions[a] != actions_to_write[a][-1]:
                actions_to_write[a].append(actions[a])

        if time.time() - write_time > 0.5:
            total_reward = 0 if steps_until_write == 0 else float(total_reward)/float(steps_until_write)
            data = {'time':(time.time()  - start_time),
                    'reward':total_reward,
                    'nfires':([f.spreading_level for f in env.components['fire']]),
                    'action':actions_to_write,
                    'waterleft':env.components['agents'][0].resources['water'],
                    'batteryleft':env.components['agents'][0].resources['battery'],                 
                    }
            log.write(None,data)
            
            total_reward = 0
            steps_until_write = 0
            write_time = time.time()

            for agent in env.components['agents']:
                actions_to_write[agent.index] = []
        else:
            total_reward += reward
            steps_until_write += 1

    env.close()

# WINDOWS SAFE PARALLEL EXECUTION
if __name__ == '__main__':
    freeze_support()
    main(args)
###
# THE END - That's all folks :)
###
