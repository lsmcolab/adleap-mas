###
# IMPORTS
###
import time
import sys
import random
import os
from tracemalloc import start
import numpy as np
from multiprocessing import freeze_support

sys.path.append(os.getcwd())
from src.envs.SmartFireBrigadeEnv import SmartFireBrigadeEnv, Agent, Fire

###
# SMART FIRE BRIGADE ENVIRONMENT SETTINGS
###
TIMEOUT = 2*60 # 5 minutes

# Main routine
def main():
    global TIMEOUT

    # Creating the environment and defining its components
    components = {
        'agents':[\
            Agent(index='A',atype='scouter',radius=30,angle=np.pi/2),
            Agent(index='B',atype='explorer',radius=30,angle=np.pi)
            ],
        'fire':[\
            Fire(position=(10,15), level=3, time_constraint=True),\
            Fire(position=(30,25), level=3, time_constraint=True),\
            Fire(position=(5,45), level=3, time_constraint=True),\
                ]
        }

    for a in components['agents']:
        try:
            os.remove('./tmp/'+a.index+'.txt')
        except:
            pass

    env = SmartFireBrigadeEnv(components=components,dim=(50,50),action_mode='control',display=True)
    state = env.reset()

    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    done = False    
    start_time = time.time()
    while not done and (time.time()  - start_time) < TIMEOUT:
        if env.display:
            env.render()

        actions = env.get_step_actions()
        if random.random()<0.5:
            actions = {'A':6,'B':6}
        state,reward,done,info = env.step(actions)
        print("Time : {} Action : {} ".format(time.time()-start_time,actions))

    env.close()

# WINDOWS SAFE PARALLEL EXECUTION
if __name__ == '__main__':
    freeze_support()
    main()
###
# THE END - That's all folks :)
###
